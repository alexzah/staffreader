"""Score an evaluation's stored readings again against corrected labels.

An evaluation keeps every staff's reading (records-*.jsonl). When a test
set's labels are corrected but its pictures are not (2026-10-15: two
players on one staff, analysis lyrics, placements out of order), the
readings need not be made again: each record's label is replaced by the
set's corrected one, staves the set no longer holds are left out, and
each is scored as `measure` scores it. A staff whose label carries masks
(test rows' 'masks': a key signature the answer does not hold as printed)
is scored without them. Writes runs/eval-<name>-fixed,
with the ladder and text results copied, and its diagnosis.

  uv run python experiments/rescore_eval.py <name> [--against <name>]
"""

import argparse
import json
import os
import shutil

from staffreader import diagnose, evaluate, measure, metrics
from staffreader.marks import strip
from staffreader.progress import Progress
from staffreader.render import CACHE


def labels() -> dict[str, tuple[list[str], list[str]]]:
    """Every corrected label and what of it is not scored (its masks), by
    the staff picture's path."""
    out = {}
    for d in (CACHE / 'data').iterdir():
        index = d / 'index.jsonl'
        if not index.exists() or not (d.name.startswith('test-') or d.name in evaluate.HELD_OUT_REAL
                                      or d.name in evaluate.HELD_OUT_RENDERED):
            continue
        for line in open(index):
            r = json.loads(line)
            out[os.path.normpath(str(d / r['image']))] = (r['tokens'], r.get('masks', []))
    return out


def rescore(rec: dict, truth: list[str], masks=()) -> dict:
    if rec.get('task') == 'notes':
        truth = strip(truth)
    r = metrics.compare(measure._score(truth), measure._score(rec['read']), frozenset(masks))
    gt_bars = [b for b in r.bars if b.gt is not None]
    return {**rec, 'truth': truth, 'tp': r.notes['core'].tp, 'fp': r.notes['core'].fp, 'fn': r.notes['core'].fn,
            'bars': len(gt_bars), 'bars_exact': sum(b.exact_full for b in gt_bars), 'errors': dict(r.errors),
            'marks': {f: [t.tp, t.fp, t.fn] for f, t in r.families.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--against')
    args = ap.parse_args()
    src, out = CACHE / 'runs' / f'eval-{args.name}', CACHE / 'runs' / f'eval-{args.name}-fixed'
    out.mkdir(parents=True, exist_ok=True)
    fixed = labels()
    files = [f for f in ('records-notes.jsonl', 'records-marks.jsonl') if (src / f).exists()]
    total = sum(1 for f in files for _ in open(src / f))
    bar = Progress(f'rescore {args.name}', total, 'staves', 'scoring stored readings against corrected labels')
    done = kept = changed = 0
    for f in files:
        with open(out / f, 'w') as w:
            for line in open(src / f):
                done += 1
                rec = json.loads(line)
                path = os.path.normpath(rec['path'])
                if path not in fixed:
                    continue  # a staff the corrected set no longer holds
                new, masks = fixed[path]
                if rec.get('unreadable'):
                    w.write(json.dumps(rec) + '\n')
                    kept += 1
                    continue
                truth = strip(new) if rec.get('task') == 'notes' else new
                if truth != rec['truth'] or masks:
                    rec = rescore(rec, new, masks)
                    changed += 1
                w.write(json.dumps(rec) + '\n')
                kept += 1
                if done % 2000 == 0:
                    bar.update(done, f'{kept} kept, {changed} rescored')
    if (src / 'ladder').exists() and not (out / 'ladder').exists():
        shutil.copytree(src / 'ladder', out / 'ladder')
    if (src / 'text.json').exists():
        shutil.copy(src / 'text.json', out / 'text.json')
    against = CACHE / 'runs' / f'eval-{args.against}-fixed' if args.against else None
    page = evaluate.report(out, against)
    bar.finish(f'{kept} of {total} records kept, {changed} rescored; diagnosis at {page}')


if __name__ == '__main__':
    main()
