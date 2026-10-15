"""E11 on the reserved test sets: v2-mix's stored readings of every test
staff (runs/eval-v2-mix-fixed, notes task), with their overfull bars read
again as triplets at the page reader's margin, scored against the same
answers. The bar length is the answer's own, as the page reader knows it
from the time signature in force. Paired bootstrap per set over staves.

  uv run python experiments/e11_test.py
Results: runs/e11-test.json."""

import collections
import json
import pathlib
import random
import sys

import cv2
import torch

from staffreader import model as M
from staffreader.baselines import staffreader as ours
from staffreader.progress import Progress
from staffreader.render import CACHE

from e11_refit import bar_length, score  # experiments/ is on the path when run from it

SYSTEMS = '--no-meter' not in sys.argv  # leave alone bars every staff of the system lengthens alike


def main():
    rows = [json.loads(line) for line in open(CACHE / 'runs' / 'eval-v2-mix-fixed' / 'records-notes.jsonl')]
    rows = [r for r in rows if not r.get('unreadable')]
    net = M.load(ours.CHECKPOINT)
    device = next(net.parameters()).device
    bar = Progress('e11 test', len(rows), 'staves', 'reserved test staves, overfull bars read again as triplets')
    systems = collections.defaultdict(list)  # test staves cut from one system: <volume>/p<page>-s<system>-...
    for r in rows:
        name = pathlib.Path(r['path'])
        systems[(str(name.parent), name.name.split('-')[0], name.name.split('-')[1])].append(r)
    out, done = [], 0
    for group in systems.values():
        limits = [bar_length(r['truth']) for r in group]
        reads = [r['read'] for r in group]
        news = reads
        if any(lim is not None and any(max(sp.voices, default=0) > lim for sp in M.bar_spans(rd, lim))
               for rd, lim in zip(reads, limits)):
            x, widths = M.batch_images([M.window(cv2.imread(r['path'], cv2.IMREAD_GRAYSCALE)) for r in group], device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                fits = net.fits(x, widths, reads, limits=limits, lead=[['task:notes']] * len(group), most=1.0,
                                skip=M.meter_changes(reads, limits) if SYSTEMS else None)
            news = [M.apply_fits(rd, f, ours.TRIPLET_MARGIN) for rd, f in zip(reads, fits)]
        for r, new in zip(group, news):
            before, after = score(r['truth'], r['read']), score(r['truth'], new)
            if before and after:
                out.append({'set': r['set'], 'changed': new != r['read'], 'before': before, 'after': after})
        done += len(group)
        bar.update(done, f"{sum(o['changed'] for o in out)} staves changed")
    rng = random.Random(0)
    table = {}
    for name in sorted({o['set'] for o in out}):
        mine = [o for o in out if o['set'] == name]

        def rate(idx, when):
            return sum(mine[i][when]['exact'] for i in idx) / max(1, sum(mine[i][when]['bars'] for i in idx))
        base = list(range(len(mine)))
        draws = sorted(rate(idx, 'after') - rate(idx, 'before')
                       for idx in ([rng.randrange(len(mine)) for _ in mine] for _ in range(1000)))
        f1 = {w: 2 * sum(o[w]['tp'] for o in mine) / max(1, sum(2 * o[w]['tp'] + o[w]['fp'] + o[w]['fn'] for o in mine))
              for w in ('before', 'after')}
        table[name] = {'staves': len(mine), 'changed': sum(o['changed'] for o in mine),
                       'bars exact before': round(rate(base, 'before'), 4), 'after': round(rate(base, 'after'), 4),
                       'interval': [round(draws[50], 4), round(draws[950], 4)],
                       'note f1 before': round(f1['before'], 4), 'note f1 after': round(f1['after'], 4)}
    (CACHE / 'runs' / ('e11-test.json' if SYSTEMS else 'e11-test-no-meter.json')).write_text(json.dumps(table, indent=1))
    bar.finish('; '.join(f"{n}: {v['bars exact before']:.1%} to {v['after']:.1%}" for n, v in table.items()))


if __name__ == '__main__':
    main()
