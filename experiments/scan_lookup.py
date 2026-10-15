"""Look up the IMSLP scans that PDMX transcriptions name, and whether the
Internet Archive's 2012 snapshot of IMSLP holds each one.

A training-data pass (2026-10-15) found PDMX scores whose metadata or
credits name the IMSLP file they were transcribed from (solo piano above
all, which no real page has trained). Pairing them needs the scans. For
each file this finds its catalog record through imslp.py (api.php only,
5 s apart; IMSLP's own lookup by number is a Special: page, which
robots.txt forbids, so the work page is searched for), then looks for the
very file, by name and size, in the Archive's snapshot. It downloads
nothing: what the snapshot holds is fetched later with archive.py, and
the rest goes to a person as a link.

  uv run python experiments/scan_lookup.py <curated.json> <out.jsonl> [--kinds k ...] [--limit N]

<curated.json> maps a kind of music to files: [{file, scores: [{title,
composer, refs...}], bars}]. Resumable: files already in <out.jsonl> are
skipped."""

import argparse
import json
import re

from staffreader import archive, imslp
from staffreader.progress import Progress

WORK = re.compile(r'imslp\.org/wiki/(?!Special:|File:)([^\s"<\\]+)')


CATALOG = re.compile(r'\b(Op|BWV|K|KV|WoO|D|Hob|HWV|RV|L|S)\.?\s*(\d+[a-z]?)', re.I)
STOP = {'von', 'van', 'der', 'den', 'de', 'des', 'du', 'la', 'le', 'and', 'for', 'the', 'no', 'nr', 'op', 'in',
        'major', 'minor', 'dur', 'moll', 'piano', 'score', 'full'}


def queries(score: dict) -> list[str]:
    """Catalog searches from a PDMX score's composer and title, most
    specific first. IMSLP's search wants every word to match, so numbers
    other than a catalog number (No. 3 of Op. 36) are left out."""
    def split(s: str) -> str:  # names run together: "BeethovenOp.31", "BartókRevised"
        return re.sub(r'([a-zà-ÿ])([A-Z])', r'\1 \2', s)
    title = split(score.get('title') or '')
    composer = split(score.get('composer') or '')
    if composer in ('', 'NA'):
        composer = re.split(r'[:\-–]', title)[0]
    composer = re.split(r'\bOp\b|\bBearbeitung\b|\bRevised\b|\barr|\d', composer, flags=re.I)[0]
    names = [w for w in re.findall(r'[A-Za-zÀ-ÿ]{3,}', composer) if w.lower() not in STOP]
    in_title = [w for w in names if w.lower() in title.lower()]
    surname = (in_title or names or [''])[-1]
    cat = CATALOG.search(title) or CATALOG.search(score.get('composer') or '')
    words = [w for w in re.findall(r'[A-Za-zÀ-ÿ]{4,}', re.sub(r'\([^)]*\)', ' ', title))
             if w.lower() not in STOP and w not in names]
    out = []
    if cat:
        out.append(f'{surname} {cat.group(1).capitalize()}.{cat.group(2)}')
    if words:
        out.append(f'{surname} {" ".join(words[:2])}')
        out.append(f'{surname} {words[0]}')
    return list(dict.fromkeys(q.strip() for q in out if q.strip()))


def find(fid: str, titles: list[str], searches: list[str], pages: int = 5) -> imslp.File | None:
    """File #[fid] on the work pages named, then on those the searches
    find, reading at most [pages] work pages."""
    tried = []
    for title in titles:
        if len(tried) >= pages:
            return None
        tried.append(title)
        if (pg := imslp.page(title)) and (f := imslp.file_on(pg, fid)):
            return f
    for q in searches:
        for title in imslp.search(q, 3):
            if title in tried:
                continue
            if len(tried) >= pages:
                return None
            tried.append(title)
            if (pg := imslp.page(title)) and (f := imslp.file_on(pg, fid)):
                return f
    return None


def lookup(entry: dict) -> dict:
    fid = str(entry['file'])
    score = entry['scores'][0]
    titles = []
    for s in entry['scores']:
        for ref in s.get('refs', []):
            if m := WORK.search(ref):
                titles.append(m.group(1).replace('_', ' '))
    rec = {'file': fid, 'kind': entry['kind'], 'bars': entry['bars'], 'title': score['title'],
           'composer': score.get('composer'), 'mxl': [s['mxl'] for s in entry['scores']],
           'link': f'https://imslp.org/wiki/Special:ImagefromIndex/{fid}'}
    f = find(fid, titles, queries(score))
    if not f:
        rec['status'] = 'not found in the catalog'
        return rec
    rec.update({'name': f.name, 'megabytes': f.megabytes, 'pages': f.pages, 'publisher': f.publisher,
                'copyright': f.copyright, 'work': f.work, 'description': f.description})
    source = 'http://imslp.org/wiki/' + f.work.replace(' ', '_')
    rec['status'] = 'not in the snapshot'
    for item in archive.search(f'collection:imslp AND source:"{source}"'):
        files = {x['name']: x for x in archive.metadata(item['identifier']).get('files', [])}
        if f.name in files:
            size = int(files[f.name].get('size', 0))
            same = f.megabytes is not None and any(abs(size / u - f.megabytes) <= 0.011 for u in (1e6, 1 << 20))
            rec['status'] = 'in the snapshot' if same else 'outdated in the snapshot'
            rec['archive'] = item['identifier']
            break
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('curated')
    ap.add_argument('out')
    ap.add_argument('--kinds', nargs='*')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--first', nargs='*', default=[], help='file numbers to look up before the rest')
    args = ap.parse_args()
    curated = json.load(open(args.curated))
    entries = [{**e, 'kind': k} for k, es in curated.items() if not args.kinds or k in args.kinds
               for e in es if not e.get('on_disk')]
    done = set()
    try:
        done = {json.loads(line)['file'] for line in open(args.out)}
    except FileNotFoundError:
        pass
    todo = [e for e in entries if str(e['file']) not in done]
    first = [str(f) for f in args.first]
    todo = sorted(todo, key=lambda e: first.index(str(e['file'])) if str(e['file']) in first else len(first))[:args.limit]
    bar = Progress('scan lookup', len(todo), 'files', 'IMSLP catalog and Archive snapshot, no downloads')
    counts = {}
    with open(args.out, 'a') as out:
        for k, e in enumerate(todo):
            bar.update(k, ', '.join(f'{n} {s}' for s, n in counts.items()))
            rec = lookup(e)
            counts[rec['status']] = counts.get(rec['status'], 0) + 1
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
    bar.finish(', '.join(f'{n} {s}' for s, n in counts.items()))


if __name__ == '__main__':
    main()
