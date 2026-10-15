"""Does real piano data lift real piano pages? Two fine-tunings of the M4
overnight model, alike in every way but one: E7a trains on what M4 trained
on, E7b adds the Lieder staves paired by content (lieder-c). Each runs a
fixed number of steps, so a busy CPU cannot shorten one more than the
other. Both are measured on held-out staves, on held-out Lieder, and on
the ladder, whose four Lieder pages come from volumes neither trains on."""
import gc, json, os, subprocess
import torch
from staffreader import train, measure
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

BASE = CACHE / 'models/m4-real/last.pt'
STEPS = 3500
MIX = ['m3-a-quarter', 'm4-a', 'real2*8']
RUNS = {'e7a-control': MIX, 'e7b-lieder': MIX + ['lieder-c*8']}
bar = Progress('experiment-lieder2', 2 * len(RUNS), 'stages',
               'E7: fine-tune M4 with and without the Lieder paired by content')
parent_of_children('experiment-lieder2')
results = {}
for k, (name, data) in enumerate(RUNS.items()):
    bar.update(2 * k, f'training {name} for {STEPS} steps')
    if not (CACHE / 'models' / name / 'last.pt').exists():
        train.train(name, data, steps=STEPS, workers=4, budget=16000, lr=1e-4, resume=str(BASE))
    bar.update(2 * k + 1, f'measuring {name}')
    ckpt = CACHE / 'models' / name / 'last.pt'
    r = measure.measure(ckpt, ['m3-a', 'm4-a', 'real2'])
    r['lieder'] = measure.measure(ckpt, ['lieder-c'], per_kind=600)['real']
    (CACHE / 'models' / name / 'measure.json').write_text(json.dumps(r, indent=1))
    # The ladder reads in its own process: hand it the memory this one cached.
    gc.collect(); torch.cuda.empty_cache()
    subprocess.run(['uv', 'run', 'omr', 'suite', '--reader', 'staffreader', '--out', str(CACHE / 'runs' / f'ladder-{name}')],
                   env={**os.environ, 'STAFFREADER_MODEL': str(ckpt)}, check=False, capture_output=True)
    results[name] = r
bar.finish('; '.join(f"{n}: real {r['real']['bars_exact']:.1%}, Lieder {r['lieder']['bars_exact']:.1%} of bars exact"
                     for n, r in results.items()))
