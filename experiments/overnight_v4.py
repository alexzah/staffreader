"""The long run after v3-mix (2026-10-16), for the author to authorize on
the decisions page. Each part is in only if its short experiment passed:

1. The text reader, from text-v3, on its own lines and, with 'words', the
   directions in words cut from engravings (r3-words-only, E16).
2. The staff reader, from v3-mix, on v3-mix's recipe and, with 'octave',
   the staves that print the octave bass clef weighed up (E15b); with
   'fingering', the keyboard scores engraved with fingering added (E15c,
   which made the reader invent fingering on real pages: not yet).
3. The new models measured on today's test sets and the ladder against
   v3-mix, whose measurement (eval-v3-mix-fixed, ladder-meter) is already
   made with today's decoder and labels, so nothing waits on a baseline.

Each stage runs in a process of its own, so one gives its memory back
before the next starts.
  uv run python experiments/overnight_v4.py [text steps] [staff hours] [words] [octave] [fingering]"""

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from staffreader import evaluate, text, train
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

TEXT_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
STAFF_HOURS = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
WORDS = 'words' in sys.argv[3:]
OCTAVE = 'octave' in sys.argv[3:]
FINGERING = 'fingering' in sys.argv[3:]
REAL = ['real3-v2-v3', 'lieder-d-v2-v3', 'orch-c-v2-v3', 'pdmx-pilot-v3', 'pdmx-a2-v3', 'pdmx-a3-v3',
        'pdmx-recover-v3', 'pdmx-b-v3', 'pdmx-b-ens-v3']
MIX = (['r2-single-v3', 'r2-grand-v3', 'r2-tenor-v3*2', 'r3-vocal-pilot-v3', 'r3-parts']
       + [f'{s}*{32 if s.startswith("pdmx-") else 8}' for s in REAL]
       + (['dose-bass-octave*4'] if OCTAVE else []) + (['r3-fingering'] if FINGERING else []))
MIX = [m for m in MIX if (CACHE / 'data' / m.split('*')[0] / 'index.jsonl').exists()]
TEXT_SETS = ['r2-single', 'r2-grand', 'r2-tenor', 'lieder-lyrics', 'r3-names', 'real-names'] + (
    ['r3-words-only'] if WORDS else [])
TEXT_WEIGHTS = {'lyrics-real': 0.4, 'name-historical': 0.6, 'name-real': 0.3}


def main():
    fresh = multiprocessing.get_context('spawn')
    bar = Progress('overnight-v4', 3, 'stages',
                   f'text reader {TEXT_STEPS} steps, staff reader {STAFF_HOURS:g} h, then v4-mix measured against v3-mix')
    parent_of_children('overnight-v4')
    bar.update(0, 'stage 1: the text reader')
    if not (CACHE / 'models/text-v4/last.pt').exists():
        with ProcessPoolExecutor(1, mp_context=fresh) as pool:
            pool.submit(text.train, 'text-v4', TEXT_SETS, steps=TEXT_STEPS, lr=1e-4,
                        resume=str(CACHE / 'models/text-v3/last.pt'), weights=TEXT_WEIGHTS).result()
    bar.update(1, 'stage 2: the staff reader')
    if not (CACHE / 'models/v4-mix/last.pt').exists():
        with ProcessPoolExecutor(1, mp_context=fresh) as pool:
            pool.submit(train.train, 'v4-mix', MIX, hours=STAFF_HOURS, workers=4, budget=16000, lr=3e-4,
                        resume=str(CACHE / 'models/v3-mix/last.pt'), tasks=True).result()
    bar.update(2, 'measuring v4-mix against v3-mix')
    os.environ['STAFFREADER_TEXT_MODEL'] = str(CACHE / 'models/text-v4/last.pt')
    with ProcessPoolExecutor(1, mp_context=fresh) as pool:
        page = pool.submit(evaluate.evaluate, CACHE / 'models/v4-mix/last.pt', 'v4-mix',
                           against=CACHE / 'runs' / 'eval-v3-mix-fixed').result()
    bar.finish(f'scorecard and diagnosis at {page}')


if __name__ == '__main__':
    main()
