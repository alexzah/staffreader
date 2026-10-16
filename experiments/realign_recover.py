"""Pair the candidate copies experiments/scan_recover.py found with the
PDMX transcriptions that named the lost IMSLP files, by content. A
candidate whose systems pair is the transcription's edition; one that
pairs none is not.

  uv run python experiments/realign_recover.py <out name> <recover.jsonl>"""

import json
import pathlib
import sys

from staffreader import realign, settings
from staffreader.baselines import staffreader as ours

PDMX = settings.path('pdmx')
rows = []
for line in open(sys.argv[2]):
    r = json.loads(line)
    for pdf in r['candidates']:
        for mxl in r['mxl']:
            rows.append({'id': pathlib.Path(mxl).stem, 'path': r['title'], 'imslp': r['file'],
                         'mxl': str(PDMX / mxl.lstrip('./')), 'pdf': pdf, 'kind': r['kind']})
print(len(rows), 'pairings of', len({r['pdf'] for r in rows}), 'candidate scans', flush=True)
realign.pair_all(rows, sys.argv[1], ours.CHECKPOINT, follow=True)
