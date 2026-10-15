"""The text reader: a picture of one line of text in, its characters out.
It reads what the staff reader leaves to it: each verse of lyrics sung
under a staff, and each instrument's name at a system's left. The same
kind of network as the staff reader, smaller: a ConvNeXt-style encoder
turns the line, HEIGHT pixels tall, into a grid two rows high, and a
transformer decoder writes the characters one at a time.

A verse is written as its syllables in order, a space between: a hyphen
after one the next continues, an underscore after one held over the
notes that follow, as data.verse_text labels them."""

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

from .marks import CHARS
from .model import Block, Encoder, LayerNorm2d, sinusoid  # noqa: F401
from .render import CACHE

HEIGHT = 64
MAX_WIDTH = 3200
MAX_CHARS = 320
VOCAB = ['<pad>', '<s>', '</s>', '�', '_'] + [c for c in CHARS if c != '_']
INDEX = {c: i for i, c in enumerate(VOCAB)}
PAD, BOS, EOS, UNKNOWN = 0, 1, 2, 3


def encode(text: str) -> list[int]:
    return [BOS] + [INDEX.get(c, UNKNOWN) for c in text[:MAX_CHARS]] + [EOS]


def decode(ids: list[int]) -> str:
    out = []
    for i in ids:
        if i == EOS:
            break
        if i > EOS:
            out.append(VOCAB[i])
    return ''.join(out)


def view(line: np.ndarray, stretch: float = 1.0) -> np.ndarray:
    """A line of text scaled to HEIGHT pixels tall."""
    zoom = HEIGHT / max(1, line.shape[0])
    width = min(MAX_WIDTH, max(32, int(round(line.shape[1] * zoom * stretch))))
    return cv2.resize(line, (width, HEIGHT), interpolation=cv2.INTER_AREA if zoom < 1 else cv2.INTER_LINEAR)


class TextReader(nn.Module):
    """A line of text read left to right: a ConvNeXt-style encoder brings
    the line, HEIGHT pixels tall, to a row of places a quarter of its width
    across, two transformer layers let each place see the line, and each
    place names a character or none. Trained with CTC, it writes the
    characters in the order they stand, so it cannot repeat or skip a
    stretch of a long verse."""

    STRIDE = 4  # pixels of the line's width to one place

    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 2):
        super().__init__()
        self.encoder = Encoder(dims=(48, 96, 192, dim), depths=(2, 2, 4, 2),
                               strides=((4, 2), (2, 2), (2, 1), (2, 1)))
        self.squeeze = nn.Linear(dim * (HEIGHT // 32), dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.1, batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, len(VOCAB))  # PAD is CTC's blank

    def forward(self, images, widths):
        """Log-probabilities of each character at each place: (B, T, C),
        and how many places each line has."""
        f = self.encoder(images)  # B, C, 2, W/4
        b, c, h, w = f.shape
        f = self.squeeze(f.permute(0, 3, 1, 2).reshape(b, w, c * h)) + sinusoid(w, c, f.device)[None]
        places = (widths + self.STRIDE - 1) // self.STRIDE
        pad = torch.arange(w, device=f.device)[None, :] >= places[:, None]
        y = self.out(self.norm(self.context(f, src_key_padding_mask=pad)))
        return y.float().log_softmax(-1), places

    @torch.no_grad()
    def read(self, images, widths) -> list[str]:
        logp, places = self(images, widths)
        best = logp.argmax(-1).tolist()
        out = []
        for row, n in zip(best, places.tolist()):
            chars, last = [], PAD
            for i in row[:n]:
                if i != last and i > EOS:
                    chars.append(VOCAB[i])
                last = i
            out.append(''.join(chars))
        return out


def batch(views: list[np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor]:
    w = max(v.shape[1] for v in views)
    x = np.full((len(views), 1, HEIGHT, w), 255, np.uint8)
    for k, v in enumerate(views):
        x[k, 0, :, :v.shape[1]] = v
    return (torch.from_numpy(x).to(device).float() / 255.0 - 0.5) / 0.5, \
        torch.tensor([v.shape[1] for v in views], device=device)


def lines(dirs: list[str], held_out: bool | None) -> list[dict]:
    """The text lines cut from renders (data/<set>/text/lines.jsonl), one
    score in fifty held out; every line of a set with held_out None, as a
    test set is read."""
    import zlib
    out = []
    for d in dirs:
        base = CACHE / 'data' / d
        f = base / 'text' / 'lines.jsonl'
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            if len(r['text']) > MAX_CHARS - 2:
                continue  # longer than the reader may write
            score = r['image'].rsplit('/', 1)[-1].split('-')[0]
            if held_out is None or (zlib.crc32(score.encode()) % 50 == 0) == held_out:
                out.append({**r, 'path': str(base / r['image'])})
    return out


def _degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """A rendered line made to look printed and scanned."""
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 18), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        _, img = cv2.threshold(img, rng.randint(120, 200), 255, cv2.THRESH_BINARY)
    return img


class Batches(torch.utils.data.IterableDataset):
    """Batches of one kind of line each, built in worker processes: each
    line read once, made to look printed and scanned, and the batch's size
    counted as padded, since every line is as wide as its widest."""

    def __init__(self, by_kind: dict[str, list[tuple[str, str]]], budget: int, seed: int,
                 weights: dict[str, float] | None = None):
        self.by_kind, self.budget, self.seed, self.weights = by_kind, budget, seed, weights or {}

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        kinds = sorted(self.by_kind)
        # Each kind as often as the others, unless [weights] says otherwise.
        odds = [self.weights.get(k, 1.0) for k in kinds]
        while True:
            pool = self.by_kind[rng.choices(kinds, odds)[0]]
            views, texts, widest = [], [], 0
            while len(views) < 256:
                path, label = rng.choice(pool)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.size == 0:
                    continue
                v = view(_degrade(img, rng), rng.uniform(0.9, 1.1))
                if views and (len(views) + 1) * max(widest, v.shape[1]) > self.budget:
                    break
                views.append(v)
                texts.append(label)
                widest = max(widest, v.shape[1])
            # Widths and lengths rounded up to a few sizes, so the GPU compiles
            # its kernels for those once rather than for every batch.
            padded = -(-widest // 256) * 256
            x = np.full((len(views), 1, HEIGHT, padded), 255, np.uint8)
            for k, v in enumerate(views):
                x[k, 0, :, :v.shape[1]] = v
            ids = [encode(t) for t in texts]
            seq = np.full((len(ids), -(-max(len(i) for i in ids) // 32) * 32 + 1), PAD, np.int64)
            for k, i in enumerate(ids):
                seq[k, :len(i)] = i
            yield torch.from_numpy(x), torch.tensor([v.shape[1] for v in views]), torch.from_numpy(seq)


def train(name: str, dirs: list[str], steps: int, lr: float = 3e-4, budget: int = 40000, resume: str | None = None,
          workers: int = 6, weights: dict[str, float] | None = None):
    """Train a text reader on [dirs]' lines for [steps] steps, a batch
    filling about [budget] pixels of padded width, and measure it on
    held-out lines. Saves models/<name>/last.pt and log.jsonl."""
    import gc
    from .progress import Progress
    device = torch.device('cuda')
    out = CACHE / 'models' / name
    out.mkdir(parents=True, exist_ok=True)
    items, held = lines(dirs, False), lines(dirs, True)
    # Names are many, short and alike; lyrics are drawn as often as names.
    by_kind: dict[str, list[tuple[str, str]]] = {}
    for r in items:
        by_kind.setdefault(r['kind'], []).append((r['path'], r['text']))
    del items
    gc.collect()
    gc.freeze()  # the workers share these lists rather than copy them
    loader = torch.utils.data.DataLoader(Batches(by_kind, budget, 0, weights), batch_size=None, num_workers=workers,
                                         prefetch_factor=4, persistent_workers=True)
    net = TextReader().to(device)
    if resume:
        net.load_state_dict(torch.load(resume, map_location=device, weights_only=True)['model'])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    bar = Progress(f'train {name}', steps, 'steps', f'training the text reader on {", ".join(dirs)}')
    log = open(out / 'log.jsonl', 'a')
    started = time.time()
    batches = iter(loader)
    for step in range(1, steps + 1):
        rate = lr * min(1.0, step / 500) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g['lr'] = rate
        x, w, seq = next(batches)
        x = (x.to(device).float() / 255.0 - 0.5) / 0.5
        w = w.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logp, places = net(x, w)
        # The characters of each line, without the start and end marks.
        target = seq[:, 1:].clone()
        lengths = ((target != PAD) & (target != EOS)).sum(1)
        flat = torch.cat([target[k, :lengths[k]] for k in range(len(target))]).to(device)
        loss = F.ctc_loss(logp.transpose(0, 1), flat, places, lengths.to(device), blank=PAD, zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps:
            log.write(json.dumps({'step': step, 'loss': round(loss.item(), 4), 'minutes': round((time.time() - started) / 60, 1)}) + '\n')
            log.flush()
            bar.update(step, f'loss {loss.item():.3f}')
        if step % 2000 == 0:
            torch.save({'model': net.state_dict(), 'step': step}, out / 'last.pt')
    torch.save({'model': net.state_dict(), 'step': steps}, out / 'last.pt')
    del batches, loader
    gc.collect()
    torch.cuda.empty_cache()
    result = measure(out / 'last.pt', held)
    (out / 'measure.json').write_text(json.dumps(result, indent=1))
    bar.finish(f"held out: {result['exact']:.1%} of lines exact, {result['cer']:.2%} of characters wrong")
    return out / 'last.pt'


def load(path, device=None) -> TextReader:
    from .model import DEVICE
    device = device or DEVICE
    net = TextReader().to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True)['model'])
    return net.eval()


def measure(path, held: list[dict], limit: int = 600) -> dict:
    """Lines read exactly, and characters wrong (edit distance over the
    true length), on held-out lines."""
    import difflib
    net = load(path)
    rng = random.Random(1)
    held = rng.sample(held, min(limit, len(held)))
    exact = errors = total = 0
    by_kind = {}
    for k in range(0, len(held), 16):
        rows = held[k:k + 16]
        x, w = batch([view(cv2.imread(r['path'], cv2.IMREAD_GRAYSCALE)) for r in rows], next(net.parameters()).device)
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            got = net.read(x, w)
        for r, g in zip(rows, got):
            sm = difflib.SequenceMatcher(None, r['text'], g)
            wrong = max(len(r['text']), len(g)) - sum(b.size for b in sm.get_matching_blocks())
            exact += g == r['text']
            errors += wrong
            total += len(r['text'])
            kd = by_kind.setdefault(r['kind'], [0, 0])
            kd[0] += g == r['text']
            kd[1] += 1
    return {'lines': len(held), 'exact': exact / max(1, len(held)), 'cer': errors / max(1, total),
            'by_kind': {k: v[0] / v[1] for k, v in by_kind.items()}}
