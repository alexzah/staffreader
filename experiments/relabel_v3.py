"""Label a set's staves with format-3 tokens (marks.vocabulary_v3), from the
MusicXML each was made from, and write DATA/<set>-v3, whose index points at
the set's own pictures. Nothing is engraved again.

An engraved staff names its exported MusicXML ('xml'); a real staff names
its transcription ('mxl': openscore:<stem>, or a path), and a staff two
players share names the second ('with'). A staff tops its system where
the set says so ('top'), or where it is the system's first staff.

  uv run python experiments/relabel_v3.py <set> [<set> ...] [--workers N] [--suffix -v3b]"""

import gzip
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

from staffreader import canonical, tokens
from staffreader.data import _parts, _root
from staffreader.progress import Progress
from staffreader.real import ANSWERS
from staffreader.render import CACHE

ORCHESTRA = CACHE / 'openscore' / 'orchestra' / 'answers'


def source(r: dict, folder) -> bytes:
    if r.get('xml'):
        path = folder / r['xml']
        return gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    mxl = r['mxl']
    if mxl.startswith('openscore:'):
        stem = mxl.split(':', 1)[1]
        mxl = ANSWERS / f'{stem}.mxl' if (ANSWERS / f'{stem}.mxl').exists() else ORCHESTRA / f'{stem}.mxl'
    return canonical.load(mxl)


def top_of(r: dict) -> bool:
    if 'top' in r:
        return bool(r['top'])
    place = re.search(r'-n(\d+)\.png$', r['image'])
    return int(place.group(1)) == 0 if place else (r['part'], r['staff']) == (0, 1)


def relabel(job) -> list[str]:
    """Format-3 index lines for one source's staves."""
    folder, rows = job
    try:
        parts = [p for p, _ in _parts(_root(source(rows[0], folder)))]
    except Exception:
        return []

    @lru_cache(maxsize=64)
    def labels(part, staff, top, with_):
        if with_ is not None:
            return tokens.from_part(tokens.merged_part(parts[part], parts[with_]), 1, top=top)
        return tokens.from_part(parts[part], staff, top=top)

    out = []
    for r in rows:
        try:
            bars, states = labels(r['part'], r['staff'], top_of(r), r.get('with'))
        except Exception:
            continue
        if r['last'] >= len(bars) or not all(s['readable'] for s in states[r['first']:r['last'] + 1]):
            continue
        image = r['image'] if r['image'].startswith('../') else f'../{folder.name}/{r["image"]}'
        out.append(json.dumps({**r, 'image': image, 'top': top_of(r),
                               'tokens': tokens.stretch(bars, states, r['first'], r['last']).tokens}))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 6
    suffix = sys.argv[sys.argv.index('--suffix') + 1] if '--suffix' in sys.argv else '-v3'
    for name in args:
        if name.startswith('--') or name.isdigit() or name == suffix:
            continue
        folder = CACHE / 'data' / name
        groups: dict[str, list[dict]] = {}
        for line in open(folder / 'index.jsonl'):
            r = json.loads(line)
            groups.setdefault(r.get('xml') or r['mxl'], []).append(r)
        out = CACHE / 'data' / f'{name}{suffix}'
        out.mkdir(parents=True, exist_ok=True)
        total = sum(len(v) for v in groups.values())
        bar = Progress(f'relabel-v3 {name}', len(groups), 'sources', f'labeling {total} staves with format-3 tokens')
        kept = 0
        with open(out / 'index.jsonl', 'w') as index, ProcessPoolExecutor(workers) as pool:
            jobs = [(folder, rows) for rows in groups.values()]
            for k, lines in enumerate(pool.map(relabel, jobs, chunksize=16), start=1):
                for line in lines:
                    index.write(line + '\n')
                kept += len(lines)
                if k % 200 == 0:
                    bar.update(k, f'{kept} staves labeled')
        bar.finish(f'{kept} of {total} staves labeled')


if __name__ == '__main__':
    main()
