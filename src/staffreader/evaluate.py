"""The loop's measuring and testing steps as commands (docs/improving.md, steps
1-5).

`evaluate` reads a model on every test set (the reserved sets whole,
held-out staves, the ladder), keeping a record of each staff, and writes
its diagnosis, against an earlier evaluation if one is named.
`experiment` trains a control and variants from one model for the same
number of steps, as a written spec says, evaluates each, and compares
every variant with the control."""

from __future__ import annotations

import gc
import json
import os
import pathlib
import subprocess

import torch
import yaml

from . import diagnose, measure
from .progress import Progress, parent_of_children
from .render import CACHE

HELD_OUT_RENDERED = ['r2-single', 'r2-grand', 'r2-tenor']
HELD_OUT_REAL = ['real3-v2', 'lieder-d-v2', 'orch-v2']


def _tasks(checkpoint: pathlib.Path) -> bool:
    return bool(torch.load(checkpoint, map_location='cpu', weights_only=True).get('tasks'))


# What a control's evaluation depends on besides its model: code that reads
# a staff (a measured record is stale when it changes), code that scores a
# reading against its label (records can be scored again without reading,
# experiments/rescore_eval.py), and the reader's decoder, which names the
# ladder's readings.
READS = ('model.py', 'staffcrop.py', 'layout.py')
SCORES = ('metrics.py', 'canonical.py', 'measure.py', 'tokens.py', 'marks.py', 'diagnose.py')


def stale(folder: pathlib.Path) -> list[str]:
    """Why an evaluation folder no longer measures what an evaluation made
    now would: its ladder read with another decoder, test labels changed
    since its records were scored, or code that reads or scores changed
    since. A control must be current before a new model is compared with
    it (docs/improving.md, "The control stays current")."""
    from .baselines import staffreader as ours
    out = []
    ladder = folder / 'ladder' / 'results.json'
    if ladder.exists():
        readers = {r['reader'] for r in json.loads(ladder.read_text()) if str(r.get('reader', '')).startswith('staffreader')}
        old = sorted(r for r in readers if not r.endswith(ours.DECODER))
        if old:
            out.append(f'its ladder was read with another decoder ({old[0]}); read it again: '
                       f'STAFFREADER_MODEL=<model> staffreader suite --reader staffreader --out {folder / "ladder"}')
    else:
        out.append('it has no ladder')
    records = folder / 'records-notes.jsonl'
    if not records.exists():
        return out + ['it has no records']
    scored = records.stat().st_mtime
    labels = [p.parent.name for p in (CACHE / 'data').glob('test-*/index.jsonl') if p.stat().st_mtime > scored]
    if labels:
        out.append(f'test labels changed since it was scored ({", ".join(sorted(labels))}); '
                   f'score it again: uv run python experiments/rescore_eval.py <name>')
    here = pathlib.Path(__file__).parent
    scoring = [f for f in SCORES if (here / f).stat().st_mtime > scored]
    if scoring:
        out.append(f'scoring code changed since it was scored ({", ".join(scoring)}); score it again '
                   f'(experiments/rescore_eval.py)')
    reading = [f for f in READS if (here / f).stat().st_mtime > scored]
    if reading:
        out.append(f'reading code changed since it was measured ({", ".join(reading)}); measure it again '
                   f'(staffreader evaluate)')
    return out


def evaluate(checkpoint: pathlib.Path, name: str, against: pathlib.Path | None = None,
             ladder: bool = True, per_kind: int = 400) -> pathlib.Path:
    """Measure [checkpoint] on every test set into runs/eval-<name>: one
    record a staff (records-notes.jsonl, and records-marks.jsonl for a
    model trained on both tasks), the ladder, and the diagnosis page.
    [against] is an earlier evaluation's folder, to compare with."""
    out = CACHE / 'runs' / f'eval-{name}'
    out.mkdir(parents=True, exist_ok=True)
    if against and (why := stale(against)):
        # Said loudly and kept with the results, not refused: a long run's
        # last stage must not be lost to it.
        warning = f'THE CONTROL {against.name} IS STALE: ' + '; '.join(why)
        print(warning, flush=True)
        (out / 'stale-control.txt').write_text(warning + '\n')
    tasks = _tasks(checkpoint)
    tests = sorted(p.name for p in (CACHE / 'data').glob('test-*') if (p / 'index.jsonl').exists())
    jobs = [('notes', 'notes' if tasks else None)] + ([('marks', 'marks')] if tasks else [])
    bar = Progress(f'evaluate {name}', len(jobs) * 3 + (1 if ladder else 0), 'stages',
                   f'measuring {checkpoint.parent.name} on every test set')
    parent_of_children(f'evaluate {name}')
    stage = 0
    for label, task in jobs:
        records = out / f'records-{label}.jsonl'
        records.unlink(missing_ok=True)
        for what, sets, kw in (('reserved test sets', tests, {'whole': True, 'per_kind': None}),
                               ('held-out rendered staves', HELD_OUT_RENDERED, {'per_kind': per_kind}),
                               ('held-out real staves', HELD_OUT_REAL, {'per_kind': per_kind})):
            bar.update(stage, f'{label}: {what}')
            present = [s for s in sets if (CACHE / 'data' / s / 'index.jsonl').exists()]
            # Each held-out set is sampled on its own, so a small set is not crowded out by a large one.
            for group in ([present] if kw.get('whole') else [[s] for s in present]):
                if group:
                    measure.measure(checkpoint, group, task=task, records=records, **kw)
            stage += 1
            gc.collect()
            torch.cuda.empty_cache()
    if ladder:
        bar.update(stage, 'the ladder')
        subprocess.run(['uv', 'run', 'staffreader', 'suite', '--reader', 'staffreader', '--out', str(out / 'ladder')],
                       env={**os.environ, 'STAFFREADER_MODEL': str(checkpoint)}, check=True)
    from . import lyrics, text
    if lyrics.TEXT_MODEL.exists():  # the text reader: rendered lines held out, and real lyric lines
        found = {}
        for label, sets, held in (('rendered lines', HELD_OUT_RENDERED, True), ('real lyric lines', ['test-lyrics'], None)):
            items = text.lines(sets, held)
            if items:
                found[label] = text.measure(lyrics.TEXT_MODEL, items)
        (out / 'text.json').write_text(json.dumps(found, indent=1))
    page = report(out, against)
    d = json.loads((out / 'diagnosis.json').read_text())
    import time
    with open(CACHE / 'runs' / 'scorecards.jsonl', 'a') as log:  # the trend across turns of the loop
        log.write(json.dumps({'when': time.strftime('%Y-%m-%d %H:%M'), 'name': name, 'model': str(checkpoint),
                              'overall': d['overall'], 'sets': d['sets'], 'ladder': d['ladder'],
                              'text': json.loads((out / 'text.json').read_text()) if (out / 'text.json').exists() else None})
                  + '\n')
    bar.finish(f'diagnosis at {page}')
    return out


def report(out: pathlib.Path, against: pathlib.Path | None = None) -> pathlib.Path:
    """The diagnosis page of an evaluation folder, against another's."""
    records = diagnose.read_records([p for p in (out / 'records-notes.jsonl', out / 'records-marks.jsonl') if p.exists()])
    # Notes are judged on the notes reading; marks on the marks reading.
    notes = [r for r in records if r.get('task') in (None, 'notes')]
    marks = [r for r in records if r.get('task') == 'marks']
    for r in notes:
        if marks:
            r['marks'] = {}
    by_path = {r['path']: r for r in marks}
    for r in notes:
        if r['path'] in by_path:
            r['marks'] = by_path[r['path']].get('marks', {})
            # The marks reading itself, for marks scored part by part (diagnose.part_marks).
            r['marks_truth'], r['marks_read'] = by_path[r['path']].get('truth'), by_path[r['path']].get('read')
    before = None
    if against:
        before = diagnose.read_records([p for p in (against / 'records-notes.jsonl',) if p.exists()])
    runs = {out.name: out / 'ladder' / 'results.json'}
    if against:
        runs[against.name] = against / 'ladder' / 'results.json'
    text = json.loads((out / 'text.json').read_text()) if (out / 'text.json').exists() else None
    return diagnose.report(notes, out, f'Diagnosis: {out.name}', against=before, ladder_runs=runs, text=text)


def experiment(spec_path: pathlib.Path) -> pathlib.Path:
    """Run a written experiment: each run fine-tuned from the spec's base
    model for its steps, evaluated, and compared with the control.

        name: e9-tuplets
        base: models/v2-mix/last.pt     # default: the reader's default model
        steps: 3500
        lr: 1.0e-4
        runs:
          control: {data: [r2-single-quarter, r2-grand, real3-v2*8], tasks: true}
          variant: {data: [r2-single-quarter, r2-grand, real3-v2*8], tasks: true,
                    boost: {tuplets: 3}, target: {tuplets: tuplets}}

    A run's boost shows chosen slices more often (train.boost_count), and
    its target names the slice it aims to improve, which the gate checks.
    """
    from . import train
    from .baselines import staffreader as ours
    spec = yaml.safe_load(spec_path.read_text())
    name, steps = spec['name'], int(spec.get('steps', 3500))
    base = CACHE / spec['base'] if spec.get('base') else ours.CHECKPOINT
    runs = spec['runs']
    if 'control' not in runs:
        raise ValueError('an experiment compares its variants with a run named control')
    order = ['control'] + [r for r in runs if r != 'control']
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    fresh = multiprocessing.get_context('spawn')
    bar = Progress(f'experiment {name}', 2 * len(order), 'stages', f'{name}: {", ".join(order)} for {steps} steps each')
    parent_of_children(f'experiment {name}')
    evals = {}
    for k, run in enumerate(order):
        cfg = runs[run]
        model = f'{name}-{run}'
        bar.update(2 * k, f'training {run}')
        # Each training and each evaluation runs in a process of its own, so
        # what one stage held is given back before the next starts: E12's
        # evaluation was killed at 16 GB in the process that had trained.
        if not (CACHE / 'models' / model / 'last.pt').exists():
            with ProcessPoolExecutor(1, mp_context=fresh) as pool:
                pool.submit(train.train, model, cfg['data'], steps=steps, workers=4, budget=16000,
                            lr=float(spec.get('lr', 1e-4)), resume=str(base), tasks=bool(cfg.get('tasks')),
                            plain=bool(cfg.get('plain')), boost=cfg.get('boost'), seed=int(cfg.get('seed', 0))).result()
        bar.update(2 * k + 1, f'evaluating {run}')
        with ProcessPoolExecutor(1, mp_context=fresh) as pool:
            evals[run] = pool.submit(evaluate, CACHE / 'models' / model / 'last.pt', model,
                                     against=evals.get('control') if run != 'control' else None).result()
    found = {run: json.loads((evals[run] / 'diagnosis.json').read_text()) for run in order}
    summary = {run: d['overall'] for run, d in found.items()}
    gate = {run: _gate(found[run]['changes'], runs[run].get('target')) for run in order if run != 'control'}
    out = CACHE / 'runs' / f'experiment-{name}.json'
    out.write_text(json.dumps({'spec': spec, 'overall': summary, 'gate': gate,
                               'pages': {run: str(evals[run] / 'index.html') for run in order}}, indent=1))
    bar.finish('; '.join(f"{run}: bars exact {100 * s['bars_exact']:.1f}%, note F1 {s['note_f1']:.3f}"
                         + (f", {gate[run]['verdict']}" if run in gate else '') for run, s in summary.items()))
    return out


def _gate(changes: list[dict], target: dict | None = None) -> dict:
    """The gate (docs/improving.md, step 6): a variant passes where nothing got
    worse than the control beyond its interval, and the slice it aims at,
    if the spec names one ({slice: value}), got better beyond it."""
    worse = [f"{c['slice']}: {c['value']} ({100 * c['change']:+.1f})" for c in changes if c['verdict'] == 'worse']
    aimed = None
    if target:
        (name, value), = target.items()
        aimed = next((c for c in changes if c['slice'] == name and c['value'] == value), None)
    hit = aimed is None or aimed['verdict'] == 'better'
    verdict = 'passes' if not worse and hit else ('fails: worse elsewhere' if worse else 'fails: its target did not improve')
    return {'verdict': verdict, 'worse': worse,
            'target': None if aimed is None else {'change': aimed['change'], 'verdict': aimed['verdict']}}
