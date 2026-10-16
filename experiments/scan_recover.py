"""Find candidate copies of scans the IMSLP catalog no longer lists under
the number a PDMX transcription names (experiments/scan_lookup.py,
status 'not found in the catalog').

Most such numbers are old uploads IMSLP has since replaced. The Internet
Archive's 2012 snapshot of IMSLP may still hold them, but it names its
files as IMSLP does (PMLP...), without the number. So the work is found
there by composer and title, its full scores are downloaded through
archive.py as candidates (SCANS/candidates/recover/IMSLP<number>/), and
pairing by content (experiments/realign_pdmx.py --candidates) decides
which, if any, is the edition the transcription was made from.

  uv run python experiments/scan_recover.py <lookup.jsonl> <out.jsonl> [--kinds k ...] [--limit N]"""

import argparse
import difflib
import json
import re

from staffreader import archive, settings
from staffreader.progress import Progress

from scan_lookup import CATALOG, STOP

SCANS = settings.path('scans')
PARTS = re.compile(r'(parts?|stimmen?|violin|viola|cello|flute|oboe|clarinet|horn|trumpet|trombone|bassoon|fagott|'
                   r'bass(?!o)|timpani|percussion|harp(?!si)|soprano|alto|tenor|organ)|_text\.pdf$', re.I)
MAX_MB = 80


def surname(composer: str, title: str) -> str:
    composer = re.sub(r'([a-zà-ÿ])([A-Z])', r'\1 \2', composer or '')
    if composer in ('', 'NA'):
        composer = re.split(r'[:\-–]', title)[0]
    composer = re.split(r'\bOp\b|\bBearbeitung\b|\bRevised\b|\barr|\d', composer, flags=re.I)[0]
    names = [w for w in re.findall(r'[A-Za-zÀ-ÿ]{3,}', composer) if w.lower() not in STOP]
    in_title = [w for w in names if w.lower() in title.lower()]
    return (in_title or names or [''])[-1]


def _fold(text: str) -> str:
    import unicodedata
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()


def candidates(r: dict, most: int = 2) -> list[tuple[str, str, int]]:
    """Up to [most] score PDFs in the snapshot for the transcription's work:
    (item, file name, bytes). Where the title names a catalog number (Op.
    13, K. 331), an item's title must carry the same one; otherwise it
    must share a word of four letters or more with the transcription's."""
    title = r['title']
    name = _fold(surname(r.get('composer') or '', title))
    if not name:
        return []
    cat = CATALOG.search(title) or CATALOG.search(r.get('composer') or '')
    words = [_fold(w) for w in re.findall(r'[A-Za-zÀ-ÿ]{4,}', re.sub(r'\([^)]*\)', ' ', title))
             if w.lower() not in STOP and _fold(w) != name]
    queries = []
    if cat:
        queries.append(f'collection:imslp AND creator:({name}) AND title:("{cat.group(1).capitalize()}.{cat.group(2)}")')
    queries += [f'collection:imslp AND creator:({name}) AND title:({w})' for w in words[:2]]
    queries.append(f'collection:imslp AND creator:({name})')
    items: list[dict] = []
    for q in queries:
        for i in archive.search(q, rows=30):
            if i['identifier'] not in {x['identifier'] for x in items}:
                items.append(i)

    def fits(item) -> bool:
        t = _fold(item.get('title') or '')
        if cat:
            return re.search(rf'\b{cat.group(1)}\.?\s*{cat.group(2)}\b', t, re.I) is not None
        return any(w[:5] in t for w in words)  # 'pianiste' finds 'Pianist'
    items = sorted((i for i in items if fits(i)),
                   key=lambda i: -difflib.SequenceMatcher(None, _fold(i.get('title') or ''), _fold(title)).ratio())
    out = []
    for it in items[:3]:
        for f in archive.metadata(it['identifier']).get('files', []):
            size = int(f.get('size', 0))
            if f['name'].lower().endswith('.pdf') and not PARTS.search(f['name']) and size <= MAX_MB * 1e6:
                out.append((it['identifier'], f['name'], size))
        if len(out) >= most:
            break
    return out[:most]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lookup')
    ap.add_argument('out')
    ap.add_argument('--kinds', nargs='*')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.lookup)]
    rows = [r for r in rows if r['status'] == 'not found in the catalog' and (not args.kinds or r['kind'] in args.kinds)]
    try:
        done = {json.loads(line)['file'] for line in open(args.out)}
    except FileNotFoundError:
        done = set()
    rows = [r for r in rows if r['file'] not in done][:args.limit]
    bar = Progress('scan recover', len(rows), 'files', 'candidate copies in the Archive snapshot, by work')
    found = 0
    with open(args.out, 'a') as out:
        for k, r in enumerate(rows):
            got = []
            for item, name, size in candidates(r):
                target = SCANS / 'candidates' / 'recover' / f"IMSLP{r['file']}" / f'{item}__{name}'
                try:
                    archive.download(item, name, target)
                    got.append(str(target))
                except Exception as e:
                    if archive.STOP_FILE.exists():
                        raise
                    print(json.dumps({'file': r['file'], 'error': repr(e)[:200]}), flush=True)
            found += bool(got)
            out.write(json.dumps({'file': r['file'], 'kind': r['kind'], 'title': r['title'], 'mxl': r['mxl'],
                                  'candidates': got}, ensure_ascii=False) + '\n')
            out.flush()
            bar.update(k + 1, f'{found} with candidates')
    bar.finish(f'{found} of {len(rows)} have candidates')


if __name__ == '__main__':
    main()
