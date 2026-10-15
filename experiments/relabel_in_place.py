"""Label a set's staves again from their transcriptions, in place, as
relabel_v2.py labels them: after a change to how scores are read (the
Hauptstimme analysis lyrics dropped, 2026-10-15). The old index is kept
beside it as index-before.jsonl.

  uv run python experiments/relabel_in_place.py <set> [<set> ...]"""

import importlib.util
import json
import pathlib
import re
import shutil
import sys

from staffreader import tokens
from staffreader.progress import Progress
from staffreader.render import CACHE

spec = importlib.util.spec_from_file_location('relabel_v2', pathlib.Path(__file__).with_name('relabel_v2.py'))
# Only its label functions are wanted; it relabels nothing without arguments.
saved, sys.argv = sys.argv, [sys.argv[0]]
relabel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relabel)
sys.argv = saved

for name in sys.argv[1:]:
    index = CACHE / 'data' / name / 'index.jsonl'
    before = index.with_name('index-before.jsonl')
    if not before.exists():
        shutil.copy(index, before)
    rows = [json.loads(l) for l in open(before)]
    bar = Progress(f'relabel {name} in place', len(rows), 'staves', 'labeling staves again from their transcriptions')
    kept = changed = 0
    with open(index, 'w') as out:
        for k, r in enumerate(sorted(rows, key=lambda r: (r['mxl'], r['part'], r['staff'])), start=1):
            place = re.search(r'-n(\d+)\.png$', r['image'])
            top = int(place.group(1)) == 0 if place else (r['part'], r['staff']) == (0, 1)
            try:
                bars, states = relabel.labels(r['mxl'], r['part'], r['staff'], top)
            except Exception:
                continue
            if r['last'] >= len(bars) or not all(s['readable'] for s in states[r['first']:r['last'] + 1]):
                continue
            new = tokens.stretch(bars, states, r['first'], r['last']).tokens
            changed += new != r['tokens']
            out.write(json.dumps({**r, 'tokens': new}) + '\n')
            kept += 1
            if k % 2000 == 0:
                bar.update(k, f'{kept} kept, {changed} changed')
    bar.finish(f'{kept} of {len(rows)} staves labeled, {changed} changed')
