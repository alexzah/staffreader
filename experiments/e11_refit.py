"""E11: read again the bars that do not add up.

E10 found that when the reader misses a triplet printed without its "3",
the bar comes out too long: on the Arabesque's real pages 30 of the 47
wrong triplet bars did not add up, and none of the 7 other wrong bars
(experiments/ladder_tuplets.py). Proposal: such a bar is read again with a
beam search that keeps only readings filling the bar (Reader.fits), and
the fit replaces the reading where the model found it at most a margin
less likely (apply_fits), so a confident reading is never forced.

This picks the margin on held-out staves only: rendered staves with their
tuplet numbers hidden that hold tuplets (exact answers for this failure),
rendered piano, and real quartet, Lieder and orchestral staves. Every
staff is read once; the fits are found once; each margin is then scored.
The reserved test sets and the ladder judge the chosen margin afterwards.

  uv run python experiments/e11_refit.py [per set]
Results: runs/e11-refit.json."""

import json
import random
import sys

import cv2
import torch

from staffreader import measure, metrics, train
from staffreader import model as M
from staffreader.baselines import staffreader as ours
from staffreader.marks import strip
from staffreader.progress import Progress
from staffreader.render import CACHE

PER = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SETS = ['r3-quiet-tuplets', 'r2-grand', 'lieder-d-v2', 'real3-v2', 'orch-v2']
MARGINS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, float('inf')]


def bar_length(truth):
    """The bar length the page reader would know for this staff: the time
    signature in force, printed earlier on the page if not here. From the
    answer, as the length most of its bars fill."""
    import collections
    seen = collections.Counter(max(sp.voices) for sp in M.bar_spans(truth) if sp.voices)
    return seen.most_common(1)[0][0] if seen else None


def score(truth, read):
    try:
        r = metrics.compare(measure._score(truth), measure._score(read))
    except Exception:
        return None
    bars = [b for b in r.bars if b.gt is not None]
    return {'bars': len(bars), 'exact': sum(b.exact_full for b in bars), 'tp': r.notes['core'].tp,
            'fp': r.notes['core'].fp, 'fn': r.notes['core'].fn}


def main():
    rng = random.Random(0)
    picks = {}
    for name in SETS:
        held = train.samples([CACHE / 'data' / name], held_out=True)
        if name == 'r3-quiet-tuplets':  # only staves that hold a tuplet
            held = [s for s in held if any(t.startswith('tup:') for t in M.decode(s.ids.tolist()))]
        picks[name] = rng.sample(held, min(PER, len(held)))
    net = M.load(ours.CHECKPOINT)
    device = next(net.parameters()).device
    bar = Progress('e11 refit', sum(len(v) for v in picks.values()), 'staves', 'reading held-out staves, then refitting bars that do not add up')
    rows, done = [], 0
    for name, items in picks.items():
        items = sorted(items, key=lambda s: s.width)
        for k in range(0, len(items), 12):
            chunk = items[k:k + 12]
            x, widths = M.batch_images([M.window(cv2.imread(s.path, cv2.IMREAD_GRAYSCALE)) for s in chunk], device)
            lead = [['task:notes']] * len(chunk)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                reads = [[t for t in r if not t.startswith('task:')] for r in net.read(x, widths, prefixes=lead)]
                fits = net.fits(x, widths, reads, limits=[bar_length(M.decode(s.ids.tolist())) for s in chunk], lead=lead,
                                most=1.0)
            for s, got, fs in zip(chunk, reads, fits):
                rows.append({'set': name, 'path': s.path, 'truth': strip(M.decode(s.ids.tolist())), 'read': got,
                             'fits': [f.__dict__ for f in fs]})
            done += len(chunk)
            bar.update(done, f'{name}: {sum(1 for r in rows if r["fits"])} staves with a fit')
    table = {}
    for name in SETS:
        mine = [r for r in rows if r['set'] == name]
        table[name] = {'staves': len(mine), 'staves with a fit': sum(1 for r in mine if r['fits']),
                       'fits': sum(len(r['fits']) for r in mine)}
        for m in MARGINS:
            t = {'bars': 0, 'exact': 0, 'tp': 0, 'fp': 0, 'fn': 0}
            for r in mine:
                fits = [M.Fit(**f) for f in r['fits']]
                got = M.apply_fits(r['read'], fits, m) if m > float('-inf') else r['read']
                sc = score(r['truth'], got)
                if sc:
                    for key in t:
                        t[key] += sc[key]
            f1 = 2 * t['tp'] / max(1, 2 * t['tp'] + t['fp'] + t['fn'])
            table[name][f'margin {m}'] = {'bars exact': round(t['exact'] / max(1, t['bars']), 4), 'note f1': round(f1, 4)}
    base = {name: {'bars exact': None} for name in SETS}
    for name in SETS:  # no refit at all
        t = {'bars': 0, 'exact': 0}
        for r in (r for r in rows if r['set'] == name):
            sc = score(r['truth'], r['read'])
            if sc:
                t['bars'] += sc['bars']
                t['exact'] += sc['exact']
        table[name]['as read'] = round(t['exact'] / max(1, t['bars']), 4)
    (CACHE / 'runs' / 'e11-refit.json').write_text(json.dumps({'table': table, 'rows': rows}, indent=1, default=str))
    bar.finish('; '.join(f"{n}: as read {v['as read']:.1%}, margin 3 {v['margin 3.0']['bars exact']:.1%}"
                         for n, v in table.items()))


if __name__ == '__main__':
    main()
