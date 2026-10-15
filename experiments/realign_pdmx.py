"""Pair scans with the PDMX transcriptions that name them, by content
(realign.py), a system at a time, as the Lieder were paired.

A scan lookup (experiments/scan_lookup.py) found which IMSLP file each
transcription says it was made from; this pairs every such scan on disk
with its transcriptions, leaving out reserved volumes (reserved.yaml) and
the test ladder's scans.

  uv run python experiments/realign_pdmx.py <out name> <curated.json> [IMSLP file id ...]
With file ids, only those scans: a pilot."""

import json
import pathlib
import sys

from staffreader import realign, settings, train
from staffreader.baselines import staffreader as ours

PDMX = settings.path('pdmx')
SCANS = settings.path('scans')

curated = json.load(open(sys.argv[2]))
only = set(sys.argv[3:])
keep = train.reserved()
rows = []
for kind, files in curated.items():
    for g in files:
        fid = str(g['file'])
        if only and fid not in only:
            continue
        pdf = next(SCANS.glob(f'IMSLP{fid}-*.pdf'), None) or next(SCANS.glob(f'IMSLP{int(fid):05d}-*.pdf'), None)
        if pdf is None or train._is_reserved(pdf.name, keep):
            continue
        for s in g['scores']:
            rows.append({'id': pathlib.Path(s['mxl']).stem, 'path': s['title'], 'imslp': pdf.name.split('-')[0][5:],
                         'mxl': str(PDMX / s['mxl'].lstrip('./')), 'kind': kind})
print(len(rows), 'transcriptions of', len({r['imslp'] for r in rows}), 'scans', flush=True)
realign.pair_all(rows, sys.argv[1], ours.CHECKPOINT)
