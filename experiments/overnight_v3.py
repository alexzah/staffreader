"""The long run for format 3 and the new real data (authorized by the author
2026-10-15, to start when E12 looks promising).

1. The text reader, from e9b-real: the lines cut from the renders, the real
   lyric lines (lieder-lyrics, a sixth of batches as in E9b), and the staff
   labels engraved as older editions print them (r3-names, kind
   'name-historical'), so labels such as 'Fg.' and 'Cor. (F)' read.
2. The staff reader, from v2-mix, both tasks: every engraved set in format 3,
   the voice parts in old C clefs, players' parts with multi-bar rests, the
   quartet and Lieder pairs, the orchestral pairs with two-player labels,
   and the scans paired with PDMX transcriptions (the notes task only,
   train.Sample.notes_only).
3. v2-mix and the new model measured on today's test sets and the ladder,
   the new one against v2-mix.

Each stage runs in a process of its own, so one gives its memory back
before the next starts.
  uv run python experiments/overnight_v3.py [text steps] [staff hours] [between]"""

import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from staffreader import evaluate, text, train
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

TEXT_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
STAFF_HOURS = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
# 'between': the sets with two-staff parts labeled with marks printed between
# the staves on both staves (-v3b), as E12's between run was, if it helped.
B = '-v3b' if 'between' in sys.argv[3:] else '-v3'
REAL = ['real3-v2-v3', f'lieder-d-v2{B}', 'orch-c-v2-v3', f'pdmx-pilot{B}', f'pdmx-a2{B}', f'pdmx-a3{B}',
        f'pdmx-recover{B}', f'pdmx-b{B}', f'pdmx-b-ens{B}']
# The scans paired with PDMX transcriptions weigh four times the other real
# sets: E14 found note F1 on real piano +0.011 at that dose over two seeds,
# where the usual weight showed nothing above chance (E13).
MIX = [m for m in ['r2-single-v3', f'r2-grand{B}', 'r2-tenor-v3*2', 'r3-vocal-pilot-v3', 'r3-parts']
       + [f'{s}*{32 if s.startswith("pdmx-") else 8}' for s in REAL]
       if (CACHE / 'data' / m.split('*')[0] / 'index.jsonl').exists()]
TEXT_SETS = ['r2-single', 'r2-grand', 'r2-tenor', 'lieder-lyrics', 'r3-names', 'real-names']
TEXT_WEIGHTS = {'lyrics-real': 0.4, 'name-historical': 0.6, 'name-real': 0.3}


def historical_kind():
    """The staff labels engraved in historical forms, as a kind of their own,
    so batches draw them as often as TEXT_WEIGHTS says."""
    f = CACHE / 'data' / 'r3-names' / 'text' / 'lines.jsonl'
    if not f.exists():
        return 0
    rows = [json.loads(line) for line in open(f)]
    for r in rows:
        if r.get('kind') == 'name':
            r['kind'] = 'name-historical'
    f.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    return sum(r['kind'] == 'name-historical' for r in rows)


def main():
    fresh = multiprocessing.get_context('spawn')
    bar = Progress('overnight-v3', 4, 'stages',
                   f'text reader {TEXT_STEPS} steps, staff reader {STAFF_HOURS:g} h, then v2-mix and v3-mix measured')
    parent_of_children('overnight-v3')
    bar.update(0, f'stage 1: the text reader ({historical_kind()} historical labels)')
    if not (CACHE / 'models/text-v3/last.pt').exists():
        with ProcessPoolExecutor(1, mp_context=fresh) as pool:
            pool.submit(text.train, 'text-v3', TEXT_SETS, steps=TEXT_STEPS, lr=1e-4,
                        resume=str(CACHE / 'models/e9b-real/last.pt'), weights=TEXT_WEIGHTS).result()
    bar.update(1, 'stage 2: the staff reader')
    if not (CACHE / 'models/v3-mix/last.pt').exists():
        with ProcessPoolExecutor(1, mp_context=fresh) as pool:
            pool.submit(train.train, 'v3-mix', MIX, hours=STAFF_HOURS, workers=4, budget=16000, lr=3e-4,
                        resume=str(CACHE / 'models/v2-mix/last.pt'), tasks=True).result()
    bar.update(2, 'measuring v2-mix on today\'s test sets')
    with ProcessPoolExecutor(1, mp_context=fresh) as pool:
        base = pool.submit(evaluate.evaluate, CACHE / 'models/v2-mix/last.pt', 'v2-mix-now').result()
    bar.update(3, 'measuring v3-mix against it')
    os.environ['STAFFREADER_TEXT_MODEL'] = str(CACHE / 'models/text-v3/last.pt')
    with ProcessPoolExecutor(1, mp_context=fresh) as pool:
        page = pool.submit(evaluate.evaluate, CACHE / 'models/v3-mix/last.pt', 'v3-mix', against=base).result()
    bar.finish(f'scorecard and diagnosis at {page}')


if __name__ == '__main__':
    main()
