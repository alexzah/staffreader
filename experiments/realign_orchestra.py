"""Pair the orchestral scans with OpenScore's transcriptions of their
movements by content (realign.py), reading with the default model.
  uv run python experiments/realign_orchestra.py <out name> [IMSLP file id ...]
With file ids, only those scans: a pilot."""
import pathlib, sys
from staffreader import realign, settings
from staffreader.baselines import staffreader as ours
from staffreader.render import CACHE

HOME = CACHE / 'openscore' / 'orchestra'
SCANS = settings.path('scans')
rows = []
for line in (HOME / 'sources.txt').read_text().splitlines():
    name, ids = line.split('|')
    for fid in ids.split():
        pdf = next(SCANS.glob(f'IMSLP{fid}-*.pdf'), None)
        if pdf is None:
            continue
        rows.append({'id': pathlib.Path(name).stem, 'path': name, 'imslp': fid,
                     'mxl': str(HOME / 'answers' / f'{pathlib.Path(name).stem}.mxl')})
only = set(sys.argv[2:])
rows = [r for r in rows if not only or r['imslp'] in only or r['imslp'].lstrip('0') in only]
print(len(rows), 'movements in', len({r['imslp'] for r in rows}), 'scans', flush=True)
realign.pair_all(rows, sys.argv[1], ours.CHECKPOINT, by_staff=True)
