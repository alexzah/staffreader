"""The overnight run, in two phases, then measured on held-out staves
and the ladder. Started only with a maintainer's approval.

1. Mixed training: every rendered staff and all four real sets (real
   works held out one in ten), real staves counted eight times.
2. A real-leaning finish from phase 1's last checkpoint: the real sets
   counted eight times against a quarter of the rendered staves (one
   staff in four), at a lower learning rate."""
import json, os, subprocess, sys
from staffreader import train, measure
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

HOURS_MIX = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
HOURS_REAL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
REAL = ['real-a*8', 'real-b*8', 'real-c*8', 'real-d*8']
bar = Progress('overnight', 3, 'phases', f'mixed {HOURS_MIX:g} h, then a real-leaning {HOURS_REAL:g} h, then measure')
parent_of_children('overnight')
bar.update(0, 'phase 1: mixed training')
train.train('night-mix', ['m3-a'] + REAL, hours=HOURS_MIX, workers=4, budget=16000, lr=5e-4)
bar.update(1, 'phase 2: real-leaning finish')
train.train('night-real', ['m3-a-quarter'] + REAL, hours=HOURS_REAL, workers=4, budget=16000, lr=1.5e-4,
            resume=str(CACHE / 'models/night-mix/last.pt'))
bar.update(2, 'measuring both phases')
results = {}
for name in ('night-mix', 'night-real'):
    ckpt = CACHE / 'models' / name / 'last.pt'
    results[name] = measure.measure(ckpt, ['m3-a', 'real-a', 'real-b', 'real-c', 'real-d'])
    (CACHE / 'models' / name / 'measure.json').write_text(json.dumps(results[name], indent=1))
    subprocess.run(['uv', 'run', 'staffreader', 'suite', '--reader', 'staffreader', '--out', str(CACHE / 'runs' / f'ladder-{name}')],
                   env={**os.environ, 'STAFFREADER_MODEL': str(ckpt)}, check=False, capture_output=True)
bar.finish('; '.join(f"{n}: real bars {r['real']['bars_exact']:.1%}, real note F1 {r['real']['note_f1_core']:.2f}"
                     for n, r in results.items()))
