"""M4's first experiment: does fine-tuning the overnight model with
rendered grand staves (piano, organ, harp; voices and all) lift the
piano pages without costing anything else? Measured against the
overnight model on held-out staves and on the ladder."""
import json, os, subprocess
from staffreader import train, measure
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

BASE = CACHE / 'models/night-real/last.pt'
HELD = ['m3-a', 'm4-a', 'real-a', 'real-b', 'real-c', 'real-d']
bar = Progress('experiments-m4', 3, 'stages', 'E5: fine-tune the overnight model with grand staves; measure it and the overnight model')
parent_of_children('experiments-m4')
bar.update(0, 'training e5-grand-staff for 7000 steps')
if not (CACHE / 'models/e5-grand-staff/last.pt').exists():
    train.train('e5-grand-staff', ['m3-a-quarter', 'm4-a', 'real-a*8', 'real-b*8', 'real-c*8', 'real-d*8'],
                steps=7000, workers=4, budget=16000, lr=2e-4, resume=str(BASE))
results = {}
for k, (name, ckpt) in enumerate((('night-real', BASE), ('e5-grand-staff', CACHE / 'models/e5-grand-staff/last.pt')), start=1):
    bar.update(k, f'measuring {name}')
    results[name] = measure.measure(ckpt, HELD)
    subprocess.run(['uv', 'run', 'staffreader', 'suite', '--reader', 'staffreader', '--out', str(CACHE / 'runs' / f'ladder-m4-{name}')],
                   env={**os.environ, 'STAFFREADER_MODEL': str(ckpt)}, check=False, capture_output=True)
(CACHE / 'runs' / 'experiments-m4.json').write_text(json.dumps(results, indent=1))
bar.finish('; '.join(f"{n}: real bars {r['real']['bars_exact']:.1%}, rendered bars {r['rendered']['bars_exact']:.1%}"
                     for n, r in results.items()))
