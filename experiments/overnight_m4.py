"""M4's overnight run, from the grand-staff experiment's model: every
rendered staff (single staves and grand staves) and the re-paired real
staves, then a real-leaning hour, then measured. Started only with the
author's authorization."""
import json, os, subprocess, sys
from staffreader import train, measure
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

HOURS_MIX = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
HOURS_REAL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
START = CACHE / 'models/e5-grand-staff/last.pt'
bar = Progress('overnight-m4', 3, 'phases', f'mixed {HOURS_MIX:g} h, then real-leaning {HOURS_REAL:g} h, then measure')
parent_of_children('overnight-m4')
bar.update(0, 'phase 1: mixed training')
train.train('m4-mix', ['m3-a', 'm4-a', 'real2*8'], hours=HOURS_MIX, workers=4, budget=16000, lr=3e-4, resume=str(START))
bar.update(1, 'phase 2: real-leaning finish')
train.train('m4-real', ['m3-a-quarter', 'm4-a', 'real2*8'], hours=HOURS_REAL, workers=4, budget=16000, lr=1e-4,
            resume=str(CACHE / 'models/m4-mix/last.pt'))
bar.update(2, 'measuring both phases')
results = {}
for name in ('m4-mix', 'm4-real'):
    ckpt = CACHE / 'models' / name / 'last.pt'
    results[name] = measure.measure(ckpt, ['m3-a', 'm4-a', 'real2'])
    (CACHE / 'models' / name / 'measure.json').write_text(json.dumps(results[name], indent=1))
    subprocess.run(['uv', 'run', 'staffreader', 'suite', '--reader', 'staffreader', '--out', str(CACHE / 'runs' / f'ladder-{name}')],
                   env={**os.environ, 'STAFFREADER_MODEL': str(ckpt)}, check=False, capture_output=True)
bar.finish('; '.join(f"{n}: real bars {r['real']['bars_exact']:.1%}, rendered bars {r['rendered']['bars_exact']:.1%}"
                     for n, r in results.items()))
