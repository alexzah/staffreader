"""Real scans to train on: the printed editions OpenScore transcribed.

OpenScore's string quartets were each transcribed from one IMSLP file,
with the edition's own system and page breaks, so every page of that
file pairs with its exact answer. This module finds, for each quartet,
that file: its edition from IMSLP's catalog (imslp.py), and a copy in the
Internet Archive's 2012 snapshot of IMSLP (archive.py), fetched within
the Archive's rules. A file the snapshot lacks, or holds at another size
than IMSLP's (IMSLP replaced it since), is listed for a person to
download by hand from IMSLP.

Only printed editions in the public domain are taken, and no work on the
test ladder."""

from __future__ import annotations

import csv
import json
import pathlib
import re

from . import archive, imslp
from .render import CACHE
from . import settings

OPENSCORE = CACHE / 'openscore'
SCANS = settings.path('scans')
# Whole works on the test ladder, never trained on.
LADDER = [r'Op\.?_?64,?_?No\.?_?5(?!\d)', r'Op\.?_?76,?_?No\.?_?3(?!\d)', r'K\.?_?465(?!\d)',
          r'Op\.?_?11,?_?No\.?_?5(?!\d)']
MANUSCRIPT = re.compile(r'manuscript|holograph|autograph', re.I)
# A part's systems are not the score's, so only full scores pair with a
# transcription's breaks.
PART = re.compile(r'violin|viola|cello|violoncell|\bparts?\b|\bpts\b|stimme', re.I)


def candidates() -> list[dict]:
    """OpenScore quartets whose source is a printed public-domain edition,
    from the catalog lookup, leaving out the test ladder's works."""
    found = max(OPENSCORE.glob('quartet-editions*.jsonl'))  # the latest lookup
    rows = [json.loads(line) for line in found.read_text().splitlines()]
    out = []
    for r in rows:
        if not r.get('publisher') or MANUSCRIPT.search(r['publisher']):
            continue
        if not (r.get('copyright') or '').startswith('Public Domain'):
            continue
        if any(re.search(p, r['path']) for p in LADDER):
            continue
        out.append(r)
    return out


def _same_size(megabytes: float | None, size: int) -> bool:
    if megabytes is None:
        return False
    return any(abs(size / unit - megabytes) <= 0.011 for unit in (1e6, 1 << 20))


def bootstrap(limit: int | None = None) -> dict:
    """Fetch each candidate's source scan from the Archive's snapshot where
    it holds the very file, and list the rest for a person to fetch."""
    got, flagged = [], []
    log = OPENSCORE / 'scans.jsonl'
    done = {json.loads(line)['imslp']: json.loads(line) for line in log.read_text().splitlines()} \
        if log.exists() else {}
    with open(log, 'a') as out:
        for r in candidates()[:limit]:
            if r['imslp'] in done:
                rec = done[r['imslp']]
            else:
                rec = {'imslp': r['imslp'], 'path': r['path'], 'publisher': r['publisher'], 'pages': r.get('pages')}
                if r.get('name'):
                    rec['name'], rec['megabytes'] = r['name'], r.get('megabytes')
                else:
                    f = imslp.find_file(r['imslp'], [r['work']])
                    rec['name'], rec['megabytes'] = (f.name, f.megabytes) if f else (None, None)
                rec['link'] = f'https://imslp.org/wiki/Special:ImagefromIndex/{r["imslp"]}'
                if PART.search(rec['name'] or ''):
                    rec['status'] = 'a part, not a score'
                    out.write(json.dumps(rec) + '\n')
                    flagged.append(rec)
                    continue
                source = 'http://imslp.org/wiki/' + r['work'].replace(' ', '_')
                items = archive.search(f'collection:imslp AND source:"{source}"')
                rec['status'] = 'not in the snapshot'
                for item in items:
                    files = {x['name']: x for x in archive.metadata(item['identifier']).get('files', [])}
                    if rec['name'] in files:
                        size = int(files[rec['name']].get('size', 0))
                        if _same_size(rec['megabytes'], size):
                            target = SCANS / f"IMSLP{r['imslp']}-{rec['name']}"
                            try:
                                archive.download(item['identifier'], rec['name'], target)
                                rec['status'], rec['file'] = 'fetched', str(target)
                            except archive.Refused as e:
                                if archive.STOP_FILE.exists():
                                    raise
                                rec['status'] = f'the Archive failed to serve it ({e})'
                        else:
                            rec['status'] = 'outdated in the snapshot'
                        break
                out.write(json.dumps(rec) + '\n')
                out.flush()
            (got if rec['status'] == 'fetched' else flagged).append(rec)
    return {'fetched': got, 'flagged': flagged}
