"""How well a checkpoint reads staves it never trained on, scored with
the harness's own metrics.

Each held-out staff is read, and its reading and its answer are written
as one-staff MusicXML and compared with metrics.compare, so the numbers
mean what the test ladder's numbers mean: note F1 at the core level
(pitch and onset), the share of bars needing no correction, and the
mistakes by kind. Rendered and real staves are scored apart."""

from __future__ import annotations

import collections
import json
import pathlib
import random

import cv2
import torch

from . import canonical, metrics, tokens
from . import model as M
from . import train
from .progress import Progress
from .render import CACHE


def _score(seq: list[str]) -> canonical.Score:
    return canonical.parse(tokens.to_musicxml([[tokens.parse(seq)]]))


def measure(checkpoint, data: list[str], per_kind: int | None = 300, batch: int = 12, seed: int = 0,
            task: str | None = None, whole: bool = False, records: pathlib.Path | None = None) -> dict:
    """A checkpoint trained with tasks reads as [task]: 'notes' for the
    notes alone, scored against the notes of each answer, or 'marks'.
    [whole] reads every staff of the sets, as a reserved test set is read,
    and per_kind None reads every staff picked. [records], if given,
    gets a line for each staff read: its answer, its reading and its
    scores, which `diagnose` slices."""
    dirs = [CACHE / 'data' / d for d in data]
    held = train.samples(dirs, held_out=None if whole else True)
    rng = random.Random(seed)
    kinds = {'rendered': [s for s in held if not s.real], 'real': [s for s in held if s.real]}
    picks = {k: rng.sample(v, min(per_kind, len(v))) if per_kind else v for k, v in kinds.items()}
    log = open(records, 'a') if records else None
    net = M.load(checkpoint)
    device = next(net.parameters()).device
    total = sum(len(v) for v in picks.values())
    name = f'{pathlib.Path(checkpoint).parent.name}/{pathlib.Path(checkpoint).stem}'
    bar = Progress(f'measure {name}', total, 'staves', f'reading held-out staves with {name}')
    out, done = {}, 0
    for kind, items in picks.items():
        tally = collections.Counter()
        errors = collections.Counter()
        families: dict[str, metrics.Tally] = {}
        items = sorted(items, key=lambda s: s.width)
        for k in range(0, len(items), batch):
            chunk = items[k:k + batch]
            views = [M.window(cv2.imread(s.path, cv2.IMREAD_GRAYSCALE)) for s in chunk]
            x, widths = M.batch_images(views, device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                lead = [[f'task:{task}']] * len(chunk) if task else None
                reads = [[t for t in r if not t.startswith('task:')] for r in net.read(x, widths, prefixes=lead)]
            for s, got in zip(chunk, reads):
                truth = M.decode(s.ids.tolist())
                if task == 'notes':
                    from .marks import strip
                    truth = strip(truth)
                tally['staves'] += 1
                tally['staves_exact'] += got == truth
                try:
                    r = metrics.compare(_score(truth), _score(got), frozenset(s.masks))
                except Exception:
                    tally['unreadable'] += 1
                    if log:
                        log.write(json.dumps({'path': s.path, 'set': _set_of(s.path), 'kind': kind, 'task': task,
                                              'truth': truth, 'read': got, 'unreadable': True}) + '\n')
                    continue
                if log:
                    gt_bars = [b for b in r.bars if b.gt is not None]
                    log.write(json.dumps({
                        'path': s.path, 'set': _set_of(s.path), 'kind': kind, 'task': task, 'truth': truth, 'read': got,
                        'tp': r.notes['core'].tp, 'fp': r.notes['core'].fp, 'fn': r.notes['core'].fn,
                        'bars': len(gt_bars), 'bars_exact': sum(b.exact_full for b in gt_bars),
                        'errors': dict(r.errors),
                        'marks': {f: [t.tp, t.fp, t.fn] for f, t in r.families.items()}}) + '\n')
                core = r.notes['core']
                tally['tp'] += core.tp
                tally['fp'] += core.fp
                tally['fn'] += core.fn
                gt_bars = [b for b in r.bars if b.gt is not None]
                tally['bars'] += len(gt_bars)
                tally['bars_exact'] += sum(b.exact_full for b in gt_bars)
                errors.update(r.errors)
                for f, t in r.families.items():
                    families.setdefault(f, metrics.Tally()).merge(t)
            done += len(chunk)
            bar.update(done, f'{kind}: {tally["staves"]} read')
        p = tally['tp'] / max(1, tally['tp'] + tally['fp'])
        rc = tally['tp'] / max(1, tally['tp'] + tally['fn'])
        out[kind] = {'staves': tally['staves'], 'staves_exact': round(tally['staves_exact'] / max(1, tally['staves']), 4),
                     'bars': tally['bars'], 'bars_exact': round(tally['bars_exact'] / max(1, tally['bars']), 4),
                     'note_f1_core': round(2 * p * rc / max(1e-9, p + rc), 4),
                     'unreadable': tally['unreadable'], 'errors': dict(errors.most_common(12)),
                     'marks': {f: {'f1': round(t.f1, 4), 'printed': t.tp + t.fn}
                               for f, t in sorted(families.items()) if t.tp + t.fn}}
    if log:
        log.close()
    bar.finish('; '.join(f"{k}: {v['bars_exact']:.1%} of bars exact, note F1 {v['note_f1_core']:.2f}"
                         for k, v in out.items() if v['staves']))
    return out


def _set_of(path: str) -> str:
    """The data set a staff was listed in: the folder after data/."""
    import re
    m = re.search(r'/data/([^/]+)/', path)
    return m.group(1) if m else ''
