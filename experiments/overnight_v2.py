"""The run that trains the reader on every mark: the text reader on the
lines cut from the renders (lyrics and instrument names), then the staff
reader on format-2 labels (the re-rendered sets, the tenor-clef renders,
and the quartet, Lieder and orchestral pairs), from the default model, as
two tasks (the notes alone, and the notes with their marks; E8c and E8d),
then both measured and the whole ladder read. Started only with the
author's authorization.
  uv run python experiments/overnight_v2.py [text steps] [staff hours]"""
import gc, json, os, subprocess, sys
import torch
from staffreader import measure, text, train
from staffreader.baselines import staffreader as ours
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

TEXT_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
STAFF_HOURS = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
START = ours.CHECKPOINT
MIX = ['r2-single', 'r2-grand', 'r2-tenor*2', 'real3-v2*8', 'lieder-d-v2*8', 'orch-v2*8']
bar = Progress('overnight-v2', 3, 'stages', f'text reader {TEXT_STEPS} steps, staff reader {STAFF_HOURS:g} h, then measure')
parent_of_children('overnight-v2')
bar.update(0, 'stage 1: the text reader')
if not (CACHE / 'models/text-a/last.pt').exists():
    text.train('text-a', ['r2-single', 'r2-grand', 'r2-tenor'], steps=TEXT_STEPS)
gc.collect(); torch.cuda.empty_cache()
bar.update(1, 'stage 2: the staff reader')
train.train('v2-mix', MIX, hours=STAFF_HOURS, workers=4, budget=16000, lr=3e-4, resume=str(START), tasks=True)
bar.update(2, 'measuring')
ckpt = CACHE / 'models/v2-mix/last.pt'
r = measure.measure(ckpt, ['m3-a', 'm4-a', 'real2'], task='notes')
r['lieder'] = measure.measure(ckpt, ['lieder-c'], per_kind=600, task='notes')['real']
r['marks-rendered'] = measure.measure(ckpt, ['r2-single', 'r2-grand'], per_kind=600, task='marks')['rendered']
r['marks-real'] = measure.measure(ckpt, ['real3-v2', 'orch-v2'], per_kind=600, task='marks')['real']
(CACHE / 'models/v2-mix/measure.json').write_text(json.dumps(r, indent=1))
gc.collect(); torch.cuda.empty_cache()
subprocess.run(['uv', 'run', 'omr', 'suite', '--reader', 'staffreader', '--out', str(CACHE / 'runs/ladder-v2-mix')],
               env={**os.environ, 'STAFFREADER_MODEL': str(ckpt)}, check=False, capture_output=True)
t = json.loads((CACHE / 'models/text-a/measure.json').read_text())
bar.finish(f"staff: real {r['real']['bars_exact']:.1%}, rendered {r['rendered']['bars_exact']:.1%} of bars exact; "
           f"text: {t['exact']:.1%} of lines exact, {t['cer']:.2%} of characters wrong")
