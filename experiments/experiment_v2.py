"""E8: training on format-2 labels. Two fine-tunings of the default model
(M4b mix), alike but for their labels and data: E8a trains on what M4b
trained on (format 1); E8b on the re-rendered sets with every mark
labeled, the quartets, Lieder and orchestral pairs relabeled, and the
tenor-clef renders. E8c (`... experiment_v2.py plain`) trains on E8b's data with
every mark taken out, to tell what the marks cost from what the data
does. Fixed steps, so a busy CPU shortens neither. Notes
are measured on the same held-out staves for both (format-1 sets), marks
on the format-2 ones, and both read the whole ladder."""
import gc, json, os, subprocess, sys
import torch
from staffreader import train, measure
from staffreader.baselines import staffreader as ours
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

BASE = ours.CHECKPOINT
STEPS = 3500
RUNS = {'e8a-control': ['m3-a-quarter', 'm4-a', 'real2*8', 'lieder-c*8'],
        'e8b-marks': ['r2-single-quarter', 'r2-grand', 'r2-tenor*2', 'real3-v2*8', 'lieder-d-v2*8', 'orch-v2*8']}
PLAIN = sys.argv[1:] == ['plain']
TASKS = sys.argv[1:] == ['tasks']
if PLAIN:
    RUNS = {'e8c-plain': RUNS['e8b-marks']}
if TASKS:  # one model for both: the notes alone, or the marks too, by the task token it starts with
    RUNS = {'e8d-tasks': RUNS['e8b-marks']}
SUFFIX = 'c' if PLAIN else ('d' if TASKS else '')
bar = Progress('experiment-v2' + SUFFIX, 2 * len(RUNS), 'stages', 'E8: fine-tune M4b on format-1 and on format-2 labels')
parent_of_children('experiment-v2' + SUFFIX)
results = {}
for k, (name, data) in enumerate(RUNS.items()):
    bar.update(2 * k, f'training {name} for {STEPS} steps')
    if not (CACHE / 'models' / name / 'last.pt').exists():
        train.train(name, data, steps=STEPS, workers=4, budget=16000, lr=1e-4, resume=str(BASE), plain=PLAIN,
                    tasks=TASKS)
    bar.update(2 * k + 1, f'measuring {name}')
    ckpt = CACHE / 'models' / name / 'last.pt'
    notes, marks = ('notes', 'marks') if TASKS else (None, None)
    r = measure.measure(ckpt, ['m3-a', 'm4-a', 'real2'], task=notes)
    r['lieder'] = measure.measure(ckpt, ['lieder-c'], per_kind=600, task=notes)['real']
    r['marks-rendered'] = measure.measure(ckpt, ['r2-single', 'r2-grand'], per_kind=600, task=marks)['rendered']
    r['marks-real'] = measure.measure(ckpt, ['real3-v2', 'orch-v2'], per_kind=600, task=marks)['real']
    (CACHE / 'models' / name / 'measure.json').write_text(json.dumps(r, indent=1))
    results[name] = r
    # The ladder reads in its own process: hand it the memory this one cached.
    gc.collect(); torch.cuda.empty_cache()
    subprocess.run(['uv', 'run', 'omr', 'suite', '--reader', 'staffreader', '--out', str(CACHE / 'runs' / f'ladder-{name}')],
                   env={**os.environ, 'STAFFREADER_MODEL': str(ckpt)}, check=False, capture_output=True)
bar.finish('; '.join(f"{n}: real {r['real']['bars_exact']:.1%}, rendered {r['rendered']['bars_exact']:.1%} of bars exact"
                     for n, r in results.items()))
