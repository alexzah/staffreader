"""Training experiments on fixed step budgets, each answering one
question, then each measured on the same held-out real works and on the
ladder, from its final checkpoint."""
import json, os, subprocess
from staffreader import train, measure
from staffreader.progress import Progress
from staffreader.render import CACHE

RUNS = [  # name, data, steps, start from, peak learning rate
    ('e1-control', ['m3-a', 'real-a*8'], 7000, None, 5e-4),
    ('e2-more-real', ['m3-a', 'real-a*8', 'real-b*8'], 7000, None, 5e-4),
    ('e3-real-x24', ['m3-a', 'real-a*24', 'real-b*24'], 7000, None, 5e-4),
    ('e4-finetune-real', ['real-a', 'real-b'], 3000, 'e1-control', 1.5e-4),
]
HELD = ['m3-a', 'real-a', 'real-b']
from staffreader.progress import parent_of_children
bar = Progress('experiments', len(RUNS) + 1, 'stages',
               'E1 control, E2 more real, E3 real x24, E4 fine-tune on real; then measure each')
parent_of_children('experiments')
for k, (name, data, steps, start, lr) in enumerate(RUNS):
    bar.update(k, f'training {name} on {data} for {steps} steps')
    if not (CACHE / 'models' / name / 'last.pt').exists() or \
            json.loads((CACHE / 'models' / name / 'run.json').read_text()).get('steps') != steps:
        train.train(name, data, steps=steps, workers=4, budget=16000, lr=lr,
                    resume=str(CACHE / 'models' / start / 'last.pt') if start else None)
bar.update(len(RUNS), 'measuring every run on held-out staves and the ladder')
results = {}
for name in [r[0] for r in RUNS]:
    ckpt = CACHE / 'models' / name / 'last.pt'
    results[name] = measure.measure(ckpt, HELD)
    (CACHE / 'models' / name / 'measure.json').write_text(json.dumps(results[name], indent=1))
    env = {**os.environ, 'STAFFREADER_MODEL': str(ckpt)}
    subprocess.run(['uv', 'run', 'staffreader', 'suite', '--reader', 'staffreader', '--out', str(CACHE / 'runs' / f'ladder-{name}')],
                   env=env, check=False, capture_output=True)
(CACHE / 'runs' / 'experiments.json').write_text(json.dumps(results, indent=1))
bar.finish('real bars exact: ' + ', '.join(f"{n} {r['real']['bars_exact']:.1%}" for n, r in results.items()))
