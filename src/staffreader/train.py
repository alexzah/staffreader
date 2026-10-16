"""Training the staff reader on staves rendered by data.py.

Each staff is seen through a window that moves and scales a little, as
the layout stage's cut does on a real page, and is then made to look
printed and scanned: uneven ink, blur, noise, specks, paper that is not
white, a slight tilt, and sometimes a hard threshold. Scores are split
into training and validation by file, so no score is in both."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from . import model as M
from .render import CACHE


def _held_out(mxl: str) -> bool:
    """Whether a score is kept for validation: one rendered score in 50,
    and one real work in 10, since real works are few."""
    every = 10 if mxl.startswith('openscore:') else 50
    return int(hashlib.sha1(mxl.encode()).hexdigest()[:4], 16) % every == 0


def _boring(seq: list[str]) -> bool:
    return all(t in ('R:measure', '|') or t.startswith(('clef:', 'key:', 'time:')) for t in seq)


class Sample:
    """One staff to learn from, kept small: hundreds of thousands of them
    sit in memory, and in every data worker's."""
    __slots__ = ('path', 'ids', 'width', 'real', 'notes_only')

    def __init__(self, path: str, ids: np.ndarray, width: int, real: bool, notes_only: bool = False):
        self.path, self.ids, self.width, self.real = path, ids, width, real
        # A real scan paired with a transcription that is not OpenScore's (a
        # PDMX user's) teaches its notes only: such transcriptions add marks
        # the print does not show, such as metronome marks for rubato.
        self.notes_only = notes_only


def reserved() -> set[str]:
    """The scans never trained on: every ladder case's, and the volumes
    reserved.yaml keeps for testing, by file name or IMSLP number."""
    import yaml
    from .cases import all_cases
    out = {t.pdf for c in all_cases() for t in c.tracks if t.kind == 'scan'}
    f = pathlib.Path(__file__).resolve().parents[2] / 'reserved.yaml'
    if f.exists():
        out |= set(str(v) for v in (yaml.safe_load(f.read_text()) or {}).get('volumes') or [])
    return out


def _is_reserved(pdf: str | None, keep: set[str]) -> bool:
    if not pdf:
        return False
    number = pdf.split('-')[0].removeprefix('IMSLP').lstrip('0')
    return pdf in keep or number in {k.removeprefix('IMSLP').lstrip('0') for k in keep if '-' not in k}


# The slices a boost may name with a number alone, and the value meant.
YES = {'voices': 'two or more', 'chords': 'chords', 'tuplets': 'tuplets', 'grace': 'grace notes'}


def boost_count(toks: list[str], boost: dict | None, rng: random.Random) -> int:
    """How many times a training staff is listed under [boost], which
    names slices as diagnose.token_features names them and how much more
    often to show them: {'tuplets': 3} shows staves with tuplets three
    times as often, {'clef': {'C4': 4}} tenor-clef staves four times. A
    staff in several boosted slices takes the largest. A number alone
    serves only the yes-or-no slices (YES); others name their value.
    Fractions are kept on average."""
    if not boost:
        return 1
    from .diagnose import token_features
    feats = token_features(toks)
    factor = 1.0
    for name, want in boost.items():
        if isinstance(want, dict):
            factor = max(factor, float(want.get(feats.get(name), 1)))
        elif name in YES and feats.get(name) == YES[name]:  # a yes-or-no slice: staves that have it
            factor = max(factor, float(want))
    whole = int(factor)
    return whole + (1 if rng.random() < factor - whole else 0)


def samples(dirs: list[pathlib.Path], held_out: bool | None, repeats: list[int] | None = None,
            plain: bool = False, boost: dict | None = None) -> list[dict]:
    """Staves to learn from (held_out False), to validate on (True), or
    every staff of a set (None), as a reserved test set is read. A staff
    of whole-bar rests alone is kept rarely in training, since parts that
    rest are common and teach little. A set's staves are listed [repeats]
    times, so a small set of real scans weighs as much as it should
    against many rendered ones."""
    from .progress import Progress
    rng = random.Random(0)
    out = []
    # No page of a scan the test ladder reads ever trains: a volume with a
    # case in it is held out whole.
    ladder = reserved()
    lines = {d: (d / 'index.jsonl').read_text().splitlines() for d in dirs}
    bar = Progress(f"load {'test' if held_out is None else 'held-out' if held_out else 'training'} staves",
                   sum(len(v) for v in lines.values()),
                   'staves', f"reading {', '.join(d.name for d in dirs)}")
    seen = 0
    for d, times in zip(dirs, repeats or [1] * len(dirs)):
        start = len(out)
        widths = _widths(d, lines[d])
        for k, line in enumerate(lines[d]):
            if k % 20000 == 0:
                bar.update(seen + k, f'{d.name}: {k} of {len(lines[d])}')
            s = json.loads(line)
            if plain:  # the notes alone, without marks, to measure what marks cost
                from .marks import strip
                s['tokens'] = strip(s['tokens'])
            if (held_out is not None and _held_out(s['mxl']) != held_out) or len(s['tokens']) + 2 > M.MAX_TOKENS:
                continue
            if held_out is False and _boring(s['tokens']) and rng.random() > 0.1:
                continue
            if not all(t in M.INDEX for t in s['tokens']):
                continue  # a note below the piano, a freak time
            if 'clef:TAB' in s['tokens']:
                continue  # six-line staves: layout and cropping take five-line ones (a crop shows its neighbours)
            if held_out is False and _is_reserved(s.get('pdf'), ladder):  # a test scan never trains
                continue
            real = s.get('font') == 'scan'
            sample = Sample(str(d / s['image']), np.array(M.encode(s['tokens']), np.int16), widths[k], real,
                            notes_only=real and not str(s.get('mxl', '')).startswith('openscore:'))
            out += [sample] * (boost_count(s['tokens'], boost, rng) if held_out is False else 1)
        seen += len(lines[d])
        if not held_out:
            out += out[start:] * (times - 1)
    bar.finish(f'{len(out)} staves')
    return out


def degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A clean staff made to look printed and scanned."""
    x = img.astype(np.float32) / 255  # paper 1, ink 0
    ink = 1 - x
    if rng.random() < 0.5:  # ink that spread or thinned
        k = np.ones((2, 2), np.uint8)
        ink = cv2.dilate(ink, k) if rng.random() < 0.5 else cv2.erode(ink, k) * 0.5 + ink * 0.5
    if rng.random() < 0.3:  # broken strokes
        holes = cv2.GaussianBlur(rng.random(ink.shape).astype(np.float32), (0, 0), rng.uniform(0.6, 1.5))
        ink = ink * (holes > rng.uniform(0.2, 0.4))
    h, w = ink.shape
    if rng.random() < 0.5:  # a slight tilt left over from straightening
        a = rng.uniform(-0.4, 0.4)
        ink = cv2.warpAffine(ink, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), (w, h))
    if rng.random() < 0.4:  # read at a lower resolution, then scaled back
        f = rng.uniform(0.5, 0.85)
        small = cv2.resize(ink, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        ink = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.6:
        ink = cv2.GaussianBlur(ink, (0, 0), rng.uniform(0.3, 1.1))
    darkness = rng.uniform(0.6, 1.0)
    paper = rng.uniform(0.75, 1.0)
    shade = cv2.resize(rng.normal(0, rng.uniform(0, 0.06), (4, 16)).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
    x = paper + shade - ink * darkness * paper
    if rng.random() < 0.4:  # specks of dirt
        n = rng.integers(1, 40)
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        for y, xx, r in zip(ys, xs, rng.uniform(0.5, 1.8, n)):
            cv2.circle(x, (int(xx), int(y)), int(round(r)), float(rng.uniform(0, 0.5)), -1)
    x = x + rng.normal(0, rng.uniform(0, 0.06), x.shape)
    if rng.random() < 0.2:  # a hard threshold, as a bilevel scan has
        x = np.where(x > rng.uniform(0.45, 0.65), 1.0, 0.0)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def with_task(ids: list[int], task: str) -> list[int]:
    """A staff's tokens as the given task writes them, after its task
    token: the notes alone, or the notes and their marks."""
    from .marks import strip
    toks = M.decode(ids)
    if task == 'notes':
        toks = strip(toks)
    return M.encode([f'task:{task}'] + toks)


class Staves(Dataset):
    def __init__(self, items: list[dict], augment: bool, tasks: bool = False, task: str | None = None):
        """With [tasks], each staff is shown as one of the two tasks, at
        random, or always as [task]."""
        self.items, self.augment, self.tasks, self.task = items, augment, tasks, task

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.items[i]
        stored = cv2.imread(s.path, cv2.IMREAD_GRAYSCALE)
        if self.augment:
            rng = np.random.default_rng()
            img = M.window(stored, spaces=M.WINDOW * rng.uniform(0.92, 1.08), shift=rng.uniform(-0.4, 0.4),
                           stretch=rng.uniform(0.85, 1.15))
            img = degrade(img, rng)
        else:
            img = M.window(stored)
        ids = s.ids.tolist()
        if self.tasks:
            task = 'notes' if s.notes_only else self.task or ('notes' if np.random.random() < 0.5 else 'marks')
            ids = with_task(ids, task)
        return img, ids


class ByWidth(Sampler):
    """Batches of staves of like width, so little of a batch is padding,
    each batch as large as the memory budget allows."""

    def __init__(self, items: list[dict], budget: int, shuffle: bool = True, seed: int = 0):
        self.widths = [self._width(s) for s in items]
        self.budget, self.shuffle, self.seed = budget, shuffle, seed
        self.epoch = 0

    @staticmethod
    def _width(s: Sample) -> int:
        return s.width

    def batches(self):
        rng = random.Random(self.seed + self.epoch)
        order = sorted(range(len(self.widths)), key=lambda i: (self.widths[i], rng.random()))
        out, cur, cur_w = [], [], 0
        for i in order:
            w = max(cur_w, self.widths[i], 256)
            if cur and w * (len(cur) + 1) > self.budget:
                out.append(cur)
                cur, w = [], max(self.widths[i], 256)
            cur.append(i)
            cur_w = w
        if cur:
            out.append(cur)
        if self.shuffle:
            rng.shuffle(out)
        return out

    def __iter__(self):
        yield from self.batches()
        self.epoch += 1

    def __len__(self):
        return len(self.batches())


def collate(batch):
    images, seqs = zip(*batch)
    t = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), t), M.PAD, dtype=torch.long)
    for k, s in enumerate(seqs):
        ids[k, :len(s)] = torch.tensor(s)
    widths = [im.shape[1] for im in images]
    w = max(32, ((max(widths) + 31) // 32) * 32)
    arr = np.zeros((len(images), 1, M.HEIGHT, w), np.float32)
    for k, im in enumerate(images):
        arr[k, 0, :, :im.shape[1]] = (255 - im.astype(np.float32)) / 255
    return torch.from_numpy(arr), torch.tensor(widths), ids


def _widths(d: pathlib.Path, lines: list[str]) -> list[int]:
    """Each staff's width as the network sees it, in index order: read from
    the pictures' headers once, then kept beside the index."""
    cache = d / 'widths.json'
    if cache.exists():
        got = json.loads(cache.read_text())
        if len(got) == len(lines):
            return got
    out = []
    for line in lines:
        h, w = _size(str(d / json.loads(line)['image']))
        out.append(min(M.MAX_WIDTH, int(round(w * M.HEIGHT / (M.WINDOW * (h / 18))))))
    cache.write_text(json.dumps(out))
    return out


def _size(path: str) -> tuple[int, int]:
    with open(path, 'rb') as f:  # a PNG's size is in its header
        head = f.read(24)
    return int.from_bytes(head[20:24], 'big'), int.from_bytes(head[16:20], 'big')


def evaluate(net: M.Reader, items: list[dict], device, limit: int = 400, read: int = 100,
             task: str | None = None) -> dict:
    """Token accuracy under teacher forcing, and the share of staves read
    exactly by greedy decoding."""
    if not items:
        return {'token_acc': 0.0, 'staff_exact': 0.0}
    net.eval()
    rng = random.Random(1)
    items = rng.sample(items, min(limit, len(items)))
    ds = Staves(items, augment=False, tasks=task is not None, task=task)
    exact = tok_right = tok_all = decoded = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for k in range(0, len(items), 16):
            imgs, seqs = zip(*(ds[i] for i in range(k, min(k + 16, len(items)))))
            x, widths = M.batch_images(list(imgs), device)
            ids = collate(list(zip(imgs, seqs)))[2].to(device)
            logits = net(x, widths, ids[:, :-1])
            pred = logits.argmax(-1)
            mask = ids[:, 1:] != M.PAD
            tok_right += ((pred == ids[:, 1:]) & mask).sum().item()
            tok_all += mask.sum().item()
            if k < read:
                lead = [[f'task:{task}']] * len(seqs) if task else None
                for got, s in zip(net.read(x, widths, prefixes=lead), seqs):
                    exact += got == M.decode(s)
                    decoded += 1
    net.train()
    return {'token_acc': round(tok_right / tok_all, 4), 'staff_exact': round(exact / max(1, decoded), 4)}


def train(name: str, data: list[str], hours: float = 3.0, lr: float = 5e-4, budget: int = 40000,
          workers: int = 10, resume: str | None = None, steps: int | None = None,
          plain: bool = False, tasks: bool = False, boost: dict | None = None, seed: int = 0) -> pathlib.Path:
    """Train a reader for [steps] steps, or for [hours], and save it to
    CACHE/models/<name>. Runs to be compared take a number of steps, so a
    busy machine slows them without giving one less training than
    another. [seed] changes the order batches are drawn in and the start of
    every random choice, so a run repeated with another seed shows how
    much two runs differ by chance alone."""
    device = 'cuda'
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = CACHE / 'models' / name
    out.mkdir(parents=True, exist_ok=True)
    names = [d.split('*')[0] for d in data]
    repeats = [int(d.split('*')[1]) if '*' in d else 1 for d in data]
    dirs = [CACHE / 'data' / d for d in names]
    train_items = samples(dirs, held_out=False, repeats=repeats, plain=plain, boost=boost)
    val_items = samples(dirs, held_out=True, plain=plain)
    real_val = [s for s in val_items if s.real]
    import gc
    gc.collect()
    gc.freeze()  # the samples live for the whole run; the collector need never walk them
    log = open(out / 'log.jsonl', 'a')
    import os
    (out / 'run.json').write_text(json.dumps({'hours': hours, 'steps': steps, 'started': time.time(),
                                              'pid': os.getpid(), 'what': f"training on {', '.join(data)}",
                                              'parent': os.environ.get('STAFFREADER_PROGRESS_PARENT')}))
    print(f'{len(train_items)} staves to learn from, {len(val_items)} held out', flush=True)
    net = M.Reader().to(device)
    if resume:
        net.load_state_dict(M.grown(torch.load(resume, map_location=device, weights_only=True)['model'], net))
    print(f'{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.98))
    sampler = ByWidth(train_items, budget, seed=seed)
    loader = DataLoader(Staves(train_items, augment=True, tasks=tasks), batch_sampler=sampler, num_workers=workers,
                        collate_fn=collate, persistent_workers=True, prefetch_factor=4)
    start, step, warmup = time.time(), 0, 2000
    deadline = start + hours * 3600 if steps is None else float('inf')
    est_steps = steps
    best = 0.0
    finished = False
    while not finished:
        for images, widths, ids in loader:
            images, widths, ids = images.to(device, non_blocking=True), widths.to(device), ids.to(device)
            elapsed = time.time() - start
            if step == 200 and steps is None:
                est_steps = int(200 * hours * 3600 / elapsed)
            frac = step / est_steps if est_steps else 0.0
            rate = lr * min(1.0, (step + 1) / warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, frac))) * 0.95 + 0.05)
            for g in opt.param_groups:
                g['lr'] = rate
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = net(images, widths, ids[:, :-1])
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                       ignore_index=M.PAD, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                log.write(json.dumps({'step': step, 'loss': round(loss.item(), 4), 'lr': rate,
                                      'minutes': round(elapsed / 60, 1)}) + '\n')
                log.flush()
            finished = time.time() >= deadline or (steps is not None and step >= steps)
            if step % 2000 == 0 or finished:
                got = evaluate(net, [s for s in val_items if not s.real], device, task='marks' if tasks else None)
                if real_val:
                    got.update({f'real_{k}': v for k, v in evaluate(net, real_val, device, task='notes' if tasks else None).items()})
                # The last line says the run is done: a timed run stops between
                # log lines, short of its full time.
                log.write(json.dumps({'step': step, 'minutes': round(elapsed / 60, 1), **got,
                                      **({'done': True} if finished else {})}) + '\n')
                log.flush()
                state = {'model': net.state_dict(), 'step': step, 'tasks': tasks, **got}
                torch.save(state, out / 'last.pt')
                if got['token_acc'] >= best:
                    best = got['token_acc']
                    torch.save(state, out / 'best.pt')
            if finished:
                break
    return out / 'best.pt'
