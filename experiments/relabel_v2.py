"""Label a set of real staves with format-2 tokens (marks included),
from the same transcriptions their pairing used. Writes DATA/<set>-v2,
whose index points at the set's own pictures.
  uv run python experiments/relabel_v2.py <set> [<set> ...]
A staff tops its system where it is the system's first staff: a pair
cut staff by staff names its place (-n0); otherwise the first part's
first staff."""
import json, re, sys
from functools import lru_cache
from staffreader import canonical, tokens
from staffreader.data import _parts, _root
from staffreader.progress import Progress
from staffreader.real import ANSWERS
from staffreader.render import CACHE

ORCHESTRA = CACHE / 'openscore' / 'orchestra' / 'answers'


@lru_cache(maxsize=64)
def parts_of(mxl: str):
    stem = mxl.split(':', 1)[1]
    path = ANSWERS / f'{stem}.mxl' if (ANSWERS / f'{stem}.mxl').exists() else ORCHESTRA / f'{stem}.mxl'
    return [p for p, _ in _parts(_root(canonical.load(path)))]


@lru_cache(maxsize=256)
def labels(mxl: str, part: int, staff: int, top: bool, with_: int | None = None):
    parts = parts_of(mxl)
    if with_ is not None:  # two players printed on one staff (realign.pair_staves)
        return tokens.from_part(tokens.merged_part(parts[part], parts[with_]), 1, top=top)
    return tokens.from_part(parts[part], staff, top=top)


for name in sys.argv[1:]:
    src = CACHE / 'data' / name
    rows = [json.loads(l) for l in open(src / 'index.jsonl')]
    rows.sort(key=lambda r: (r['mxl'], r['part'], r['staff']))  # so the caches hold
    out = CACHE / 'data' / f'{name}-v2'
    out.mkdir(parents=True, exist_ok=True)
    bar = Progress(f'relabel {name}', len(rows), 'staves', 'labeling real staves with format-2 tokens')
    kept = 0
    with open(out / 'index.jsonl', 'w') as index:
        for k, r in enumerate(rows, start=1):
            place = re.search(r'-n(\d+)\.png$', r['image'])
            top = int(place.group(1)) == 0 if place else (r['part'], r['staff']) == (0, 1)
            try:
                bars, states = labels(r['mxl'], r['part'], r['staff'], top, r.get('with'))
            except Exception:
                continue
            if r['last'] >= len(bars) or not all(s['readable'] for s in states[r['first']:r['last'] + 1]):
                continue
            image = r['image'] if r['image'].startswith('../') else f'../{name}/{r["image"]}'
            index.write(json.dumps({**r, 'image': image, 'top': top,
                                    'tokens': tokens.stretch(bars, states, r['first'], r['last']).tokens}) + '\n')
            kept += 1
            if k % 2000 == 0:
                bar.update(k, f'{kept} kept')
    bar.finish(f'{kept} of {len(rows)} staves labeled')
