"""M4b's overnight run: the M4 recipe again, from the M4 model, with the
Lieder staves paired by content (lieder-c) added to both phases, as E7
tested in 30 minutes. Started only with a maintainer's approval.
  uv run python experiments/overnight_m4b.py [hours mixed] [hours real]"""
import gc, json, os, subprocess, sys
import torch
from staffreader import train, measure
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

HOURS_MIX = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
HOURS_REAL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
START = CACHE / 'models/m4-real/last.pt'
bar = Progress('overnight-m4b', 3, 'phases', f'mixed {HOURS_MIX:g} h, then real-leaning {HOURS_REAL:g} h, then measure')
parent_of_children('overnight-m4b')
bar.update(0, 'phase 1: mixed training')
train.train('m4b-mix', ['m3-a', 'm4-a', 'real2*8', 'lieder-c*8'], hours=HOURS_MIX, workers=4, budget=16000, lr=3e-4,
            resume=str(START))
bar.update(1, 'phase 2: real-leaning finish')
train.train('m4b-real', ['m3-a-quarter', 'm4-a', 'real2*8', 'lieder-c*8'], hours=HOURS_REAL, workers=4, budget=16000,
            lr=1e-4, resume=str(CACHE / 'models/m4b-mix/last.pt'))
bar.update(2, 'measuring both phases')
results = {}
for name in ('m4b-mix', 'm4b-real'):
    ckpt = CACHE / 'models' / name / 'last.pt'
    r = measure.measure(ckpt, ['m3-a', 'm4-a', 'real2'])
    r['lieder'] = measure.measure(ckpt, ['lieder-c'], per_kind=600)['real']
    (CACHE / 'models' / name / 'measure.json').write_text(json.dumps(r, indent=1))
    results[name] = r
    # The ladder reads in its own process: hand it the memory this one cached.
    gc.collect(); torch.cuda.empty_cache()
    subprocess.run(['uv', 'run', 'omr', 'suite', '--reader', 'staffreader', '--out', str(CACHE / 'runs' / f'ladder-{name}')],
                   env={**os.environ, 'STAFFREADER_MODEL': str(ckpt)}, check=False, capture_output=True)
bar.finish('; '.join(f"{n}: real {r['real']['bars_exact']:.1%}, Lieder {r['lieder']['bars_exact']:.1%} of bars exact"
                     for n, r in results.items()))
