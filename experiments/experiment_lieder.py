"""Does real piano data lift real piano pages? The M4 overnight model,
fine-tuned for 7,000 steps with the confirmed Lieder staves, measured
against itself on held-out staves and the ladder."""
import gc, json, os, subprocess
import torch
from staffreader import train, measure
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

BASE = CACHE / 'models/m4-real/last.pt'
bar = Progress('experiment-lieder', 2, 'stages', 'E6: fine-tune the M4 overnight model with confirmed real Lieder staves')
parent_of_children('experiment-lieder')
bar.update(0, 'training e6-lieder for 7000 steps')
if not (CACHE / 'models/e6-lieder/last.pt').exists():
    train.train('e6-lieder', ['m3-a-quarter', 'm4-a', 'real2*8', 'lieder-ok*8'], steps=7000, workers=4, budget=16000,
                lr=1e-4, resume=str(BASE))
bar.update(1, 'measuring e6-lieder')
ckpt = CACHE / 'models/e6-lieder/last.pt'
r = measure.measure(ckpt, ['m3-a', 'm4-a', 'real2'])
(CACHE / 'models/e6-lieder/measure.json').write_text(json.dumps(r, indent=1))
# The ladder reads in its own process: hand it the memory this one cached.
gc.collect(); torch.cuda.empty_cache()
subprocess.run(['uv', 'run', 'omr', 'suite', '--reader', 'staffreader', '--out', str(CACHE / 'runs/ladder-e6-lieder')],
               env={**os.environ, 'STAFFREADER_MODEL': str(ckpt)}, check=False, capture_output=True)
bar.finish(f"real bars {r['real']['bars_exact']:.1%}, rendered bars {r['rendered']['bars_exact']:.1%}")
