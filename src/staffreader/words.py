"""Directions in words found on a page: where they are printed.

The staff reader spells directions in words (dolce, a 2., Allegro vivace)
a character at a time inside a staff's crop, and gets most wrong, some
made up. The text reader reads them well once it is shown where they are
(E16). This finds them: a small network marks which of a page's pixels
are such words, taught on engraved pages whose words' boxes the engraver
gives (data._keep_page), with lyrics and staff labels as text it passes
over. The marked regions become boxes for the text reader.

  finder(): the network.  train(): teach it.  find(): boxes on a page."""

from __future__ import annotations

import json
import math
import pathlib
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .render import CACHE

SPACE = 8.0  # pixels between staff lines the finder sees a page at
TILE = 384


class Finder(nn.Module):
    """A small U-Net: a page's pixels in, the chance each is a word's out."""

    def __init__(self, width: int = 24):
        super().__init__()
        w = [width, 2 * width, 4 * width, 8 * width]

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.down = nn.ModuleList([block(1, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3])])
        self.up = nn.ModuleList([block(w[3] + w[2], w[2]), block(w[2] + w[1], w[1]), block(w[1] + w[0], w[0])])
        self.out = nn.Conv2d(w[0], 1, 1)

    def forward(self, x):
        skips = []
        for k, d in enumerate(self.down):
            x = d(x)
            if k < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for u in self.up:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = u(torch.cat([x, skips.pop()], 1))
        return self.out(x)


def _scaled(gray: np.ndarray, space: float) -> tuple[np.ndarray, float]:
    s = SPACE / space
    return cv2.resize(gray, (max(1, round(gray.shape[1] * s)), max(1, round(gray.shape[0] * s))),
                      interpolation=cv2.INTER_AREA), s


class Pages(torch.utils.data.IterableDataset):
    """Tiles of engraved pages and the mask of their words."""

    def __init__(self, rows: list[dict], base: pathlib.Path, seed: int = 0):
        self.rows, self.base, self.seed = rows, base, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0))
        while True:
            r = rng.choice(self.rows)
            gray = cv2.imread(str(self.base / r['image']), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            img, s = _scaled(gray, r['space'] * rng.uniform(0.85, 1.15))
            mask = np.zeros(img.shape, np.float32)
            for b in r['words']:
                cv2.rectangle(mask, (int(b[0] * s), int(b[1] * s)), (int(math.ceil(b[2] * s)), int(math.ceil(b[3] * s))), 1.0, -1)
            h, w = img.shape
            if h < TILE or w < TILE:
                img = cv2.copyMakeBorder(img, 0, max(0, TILE - h), 0, max(0, TILE - w), cv2.BORDER_CONSTANT, value=255)
                mask = cv2.copyMakeBorder(mask, 0, max(0, TILE - h), 0, max(0, TILE - w), cv2.BORDER_CONSTANT, value=0)
                h, w = img.shape
            # Tiles with words in them half the time: words are a small share of a page.
            words = np.argwhere(mask > 0)
            if len(words) and rng.random() < 0.5:
                cy, cx = words[rng.randrange(len(words))]
                y = min(max(0, cy - rng.randrange(TILE)), h - TILE)
                x = min(max(0, cx - rng.randrange(TILE)), w - TILE)
            else:
                y, x = rng.randrange(h - TILE + 1), rng.randrange(w - TILE + 1)
            tile = img[y:y + TILE, x:x + TILE].astype(np.float32)
            # A page made to look printed: contrast, blur and noise.
            tile = tile * rng.uniform(0.8, 1.0) + rng.uniform(0, 30)
            if rng.random() < 0.5:
                tile = cv2.GaussianBlur(tile, (3, 3), rng.uniform(0.3, 1.0))
            tile = np.clip(tile + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 6, tile.shape), 0, 255)
            yield (torch.from_numpy((255 - tile) / 255.0).unsqueeze(0).float(),
                   torch.from_numpy(mask[y:y + TILE, x:x + TILE]).unsqueeze(0))


def rows(sets: list[str], held_out: bool) -> list[tuple[pathlib.Path, dict]]:
    """Each engraved page of [sets] and its words' boxes; one score in twenty held out."""
    import zlib
    out = []
    for name in sets:
        base = CACHE / 'data' / name
        f = base / 'pages' / 'boxes.jsonl'
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            score = pathlib.Path(r['image']).name.split('-')[0]
            if (zlib.crc32(score.encode()) % 20 == 0) == held_out:
                out.append((base, r))
    return out


def train(name: str, sets: list[str], steps: int = 6000, batch: int = 12, lr: float = 1e-3, workers: int = 4) -> pathlib.Path:
    """Teach a finder on [sets]' engraved pages; saves models/<name>/last.pt."""
    from .progress import Progress
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = CACHE / 'models' / name
    out.mkdir(parents=True, exist_ok=True)
    found = rows(sets, False)
    base = found[0][0]
    data = Pages([r for _, r in found], base)
    loader = torch.utils.data.DataLoader(data, batch_size=batch, num_workers=workers)
    net = Finder().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    bar = Progress(f'train {name}', steps, 'steps', f'the word finder on {len(found)} engraved pages')
    started, it = time.time(), iter(loader)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g['lr'] = lr * min(1.0, step / 300) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        logit = net(x)
        # Words are few: a word's pixel weighs as much as five of paper.
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(5.0, device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps:
            bar.update(step, f'loss {loss.item():.4f}, {(time.time() - started) / 60:.1f} min')
        if step % 2000 == 0 or step == steps:
            torch.save({'model': net.state_dict(), 'step': step}, out / 'last.pt')
    bar.finish(f'loss {loss.item():.4f}')
    return out / 'last.pt'


def load(path: pathlib.Path, device=None) -> Finder:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Finder().to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True)['model'])
    return net.eval()


@torch.no_grad()
def find(net: Finder, gray: np.ndarray, space: float, cut: float = 0.5) -> list[tuple[int, int, int, int]]:
    """The boxes of the words a page prints, in its own pixels."""
    img, s = _scaled(gray, space)
    h, w = img.shape
    H, W = math.ceil(h / 8) * 8, math.ceil(w / 8) * 8
    padded = cv2.copyMakeBorder(img, 0, H - h, 0, W - w, cv2.BORDER_CONSTANT, value=255)
    device = next(net.parameters()).device
    x = torch.from_numpy((255 - padded.astype(np.float32)) / 255.0)[None, None].to(device)
    prob = torch.sigmoid(net(x))[0, 0, :h, :w].float().cpu().numpy()
    mask = (prob > cut).astype(np.uint8)
    # A word's letters close into one region; separate words stay apart.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for k in range(1, n):
        x0, y0, bw, bh, area = stats[k]
        if area < 12 or bh < 0.6 * SPACE:
            continue
        out.append((int(x0 / s), int(y0 / s), int((x0 + bw) / s), int((y0 + bh) / s)))
    return out


DYNAMIC = {'p', 'pp', 'ppp', 'f', 'ff', 'fff', 'mf', 'mp', 'sf', 'sfz', 'fz', 'fp', 'rf', 'rfz', 'sff', 'sffz'}


def _drop_words(bar) -> None:
    """A bar's directions in words, as the staff reader spelled them, taken out."""
    def keep(toks):
        out, inside = [], False
        for t in toks:
            if t == 'w(':
                inside = True
            elif inside:
                inside = t != ')'
            else:
                out.append(t)
        return out
    for n in bar.notes:
        if n.get('before'):
            n['before'] = keep(n['before'])
    bar.after = keep(bar.after)


def place_words(found: list[tuple[tuple[int, int, int, int], str]], staves: list, bars: list[list], ends: list[float],
                replace_all: bool = False) -> int:
    """Put the words [found] (each its box on the page and the text read
    in it) on the staves of one system: [staves] are the layout's staves,
    [bars] each staff's bars as read (notes placed by positions.place),
    [ends] the system's barlines. A word goes to the staff it stands
    nearest, the bar it stands in, and the first note at or after its left
    edge; the staff reader's own words in that bar give way to it. With
    [replace_all], the staff reader's words in every bar give way, found
    or not. Returns how many words were placed."""
    from . import marks
    if replace_all:
        for staff_bars in bars:
            for bar in staff_bars:
                _drop_words(bar)
    starts = [min(st.x0 for st in staves)] + list(ends[:-1])
    placed, cleared = 0, set()
    for (x0, y0, x1, y1), text in sorted(found, key=lambda f: f[0][0]):
        text = ' '.join(text.split())
        if not marks.readable(text) or text.lower().strip('.') in DYNAMIC or len(text) < 2:
            continue
        cy = (y0 + y1) / 2
        k = min(range(len(staves)), key=lambda i: 0 if staves[i].top <= cy <= staves[i].bottom
                else min(abs(cy - staves[i].top), abs(cy - staves[i].bottom)))
        b = next((i for i, e in enumerate(ends) if x0 < e), len(ends) - 1)
        if b >= len(bars[k]):
            continue
        bar = bars[k][b]
        if (k, b) not in cleared:
            _drop_words(bar)
            cleared.add((k, b))
        sp = staves[k].space
        notes = [n for n in bar.notes if not n.get('chord')]
        if not notes:
            bar.after = marks.text_tokens('w(', text) + bar.after
            placed += 1
            continue
        at = [n for n in notes if 'x' in n and n['x'] >= x0 - 0.5 * sp]
        note = at[0] if at else (notes[-1] if any('x' in n for n in notes) else notes[0])
        if not at and x0 > starts[b] and any('x' in n for n in notes) and x0 > max(n['x'] for n in notes if 'x' in n):
            bar.after = marks.text_tokens('w(', text) + bar.after
        else:
            note['before'] = marks.text_tokens('w(', text) + (note.get('before') or [])
        placed += 1
    return placed
