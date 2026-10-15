"""Fetch the scans a scan lookup found (experiments/scan_lookup.py) through
the sources waterfall (sources.find): files already here, Commons, the
Internet Archive's 2012 snapshot of IMSLP, and so on; whatever none of
them holds is left for a person, as a link. Only printed editions in the
public domain in the US, served by imslp.org itself, are taken.

  uv run python experiments/scan_fetch.py <lookup.jsonl> [--status 'in the snapshot']"""

import argparse
import json
import re

from staffreader import sources
from staffreader.progress import Progress


def wanted(r: dict) -> bool:
    c, d, p = r.get('copyright') or '', r.get('description') or '', r.get('publisher') or ''
    return (c.startswith('Public Domain') and 'Non-PD US' not in c and '(EU)' not in d
            and not re.search(r'manuscript|autograph|holograph', p, re.I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lookup')
    ap.add_argument('--status', default='in the snapshot')
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.lookup)]
    rows = [r for r in rows if r['status'] == args.status and wanted(r)]
    bar = Progress('scan fetch', len(rows), 'scans', 'fetching found scans through the sources waterfall')
    for k, r in enumerate(rows):
        composer = re.search(r'\(([^()]*)\)\s*$', r['work'] or '')
        w = sources.Want(imslp=r['file'], name=r['name'], megabytes=r['megabytes'], pages=r['pages'], work=r['work'],
                         composer=composer.group(1) if composer else '', publisher=r['publisher'])
        got = sources.find(w)
        print(json.dumps({'file': r['file'], 'source': got.source, 'path': got.path, 'notes': got.notes}), flush=True)
        bar.update(k + 1, r['work'][:50])
    bar.finish(f'{len(rows)} looked for')


if __name__ == '__main__':
    main()
