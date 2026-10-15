"""Pair the Lieder volumes on disk with their songs by content
(realign.py), reading with the M4 model, which never trained on Lieder.
  uv run python experiments/realign_lieder.py <out name> [IMSLP file id ...]
With file ids, only those volumes: a pilot."""
import json, sys
import yaml
from staffreader import realign
from staffreader.render import CACHE

H = CACHE / 'openscore'
songs = yaml.safe_load(open(H / 'lieder-scores-pinned.yaml'))
by_path = {s['path']: str(sid) for sid, s in songs.items()}
volumes = {}
for s in ('lieder-a', 'lieder-b'):
    for r in json.load(open(CACHE / 'data' / s / 'report.json')):
        if 'pdf' in r and r.get('work') in by_path:
            volumes.setdefault(r['pdf'].split('-')[0][5:], set()).add(r['work'])
only = set(sys.argv[2:])
rows = [{'id': by_path[w], 'path': w, 'imslp': fid, 'collection': 'lieder'}
        for fid, works in volumes.items() if not only or fid in only for w in sorted(works)]
print(len(rows), 'songs in', len({r['imslp'] for r in rows}), 'volumes', flush=True)
from staffreader.baselines import staffreader as ours
realign.pair_all(rows, sys.argv[1], ours.CHECKPOINT)
