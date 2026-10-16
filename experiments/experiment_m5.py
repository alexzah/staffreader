"""E8: does M5's data lift the reader? Two fine-tunings of the default
model (M4b mix), alike but for the data: E8a trains on what M4b trained
on; E8b on the quartets and Lieder re-paired with flattened pages
(real3, lieder-d), the tenor-clef renders (m5-tenor) and the orchestral
pairs (orch-a). Fixed steps, so a busy CPU shortens neither. Both are
measured on the same held-out staves (the quartets as real2 holds them
out, the Lieder as lieder-c does, the orchestral pairs, the tenor-clef
renders) and on the whole ladder, whose pages no run trains on."""
import gc, json, os, subprocess
import torch
from staffreader import train, measure
from staffreader.baselines import staffreader as ours
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

BASE = ours.CHECKPOINT
STEPS = 3500
RUNS = {'e8a-control': ['m3-a-quarter', 'm4-a', 'real2*8', 'lieder-c*8'],
        'e8b-m5': ['m3-a-quarter', 'm4-a', 'real3*8', 'lieder-d*8', 'm5-tenor*2', 'orch-a*8']}
bar = Progress('experiment-m5', 2 * len(RUNS), 'stages', 'E8: fine-tune M4b with and without the M5 data')
parent_of_children('experiment-m5')
results = {}
for k, (name, data) in enumerate(RUNS.items()):
    bar.update(2 * k, f'training {name} for {STEPS} steps')
    if not (CACHE / 'models' / name / 'last.pt').exists():
        train.train(name, data, steps=STEPS, workers=4, budget=16000, lr=1e-4, resume=str(BASE))
    bar.update(2 * k + 1, f'measuring {name}')
    ckpt = CACHE / 'models' / name / 'last.pt'
    r = measure.measure(ckpt, ['m3-a', 'm4-a', 'real2'])
    r['lieder'] = measure.measure(ckpt, ['lieder-c'], per_kind=600)['real']
    r['orchestra'] = measure.measure(ckpt, ['orch-a'], per_kind=600)['real']
    r['tenor'] = measure.measure(ckpt, ['m5-tenor'], per_kind=300)['rendered']
    (CACHE / 'models' / name / 'measure.json').write_text(json.dumps(r, indent=1))
    results[name] = r
    # The ladder reads in its own process: hand it the memory this one cached.
    gc.collect(); torch.cuda.empty_cache()
    subprocess.run(['uv', 'run', 'staffreader', 'suite', '--reader', 'staffreader', '--out', str(CACHE / 'runs' / f'ladder-{name}')],
                   env={**os.environ, 'STAFFREADER_MODEL': str(ckpt)}, check=False, capture_output=True)
bar.finish('; '.join(f"{n}: real {r['real']['bars_exact']:.1%}, Lieder {r['lieder']['bars_exact']:.1%}, "
                     f"orchestra {r['orchestra']['bars_exact']:.1%}, tenor {r['tenor']['bars_exact']:.1%}"
                     for n, r in results.items()))
