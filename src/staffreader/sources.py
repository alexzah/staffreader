"""Where to find a scan: every public source in turn, within each one's
rules, and a person's hand only when all of them fail.

A scan is wanted as an IMSLP file: its number, its name and size on
IMSLP, its page count, and the work and edition it holds. The sources
are tried in this order, each through the one module that keeps its
rules:

1. here: the file is already in SCANS, as IMSLP named it;
2. Wikimedia Commons: a copy of the same IMSLP file, fetched page by
   page as Commons draws them. Commons is reached only through a polite
   Wikimedia client that the wikimedia setting names as a command, with
   search, info and fetch subcommands, and is skipped where none is
   named;
3. the Internet Archive's 2012 snapshot of IMSLP (archive.py): the same
   file at the same size;
4. other scans on the Internet Archive (archive.py): a library's scan of
   the same edition. Such a scan is not the same file, so it is kept only
   as a candidate, under SCANS/candidates, until pairing with the
   transcription shows its systems are the edition's;
5. the Bavarian State Library (mdz.py): a public-domain copy of the same
   publisher's print, also a candidate until it pairs;
6. a person: a link to download the file by hand from IMSLP, which
   forbids scripted downloads (imslp.py).

IMSLP's mirrors in Europe, Canada and Asia are run by IMSLP's own
organization, so they are treated as IMSLP and never scripted."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass, field

from . import archive, mdz
from . import settings

SCANS = settings.path('scans')
MAX_CANDIDATE_MB = 60


@dataclass
class Want:
    imslp: str  # the file's number on IMSLP
    name: str | None  # its name on IMSLP, as File:<name>
    megabytes: float | None
    pages: int | None
    work: str  # its IMSLP work page
    composer: str
    publisher: str | None = None


@dataclass
class Found:
    want: Want
    source: str  # 'here', 'commons', 'archive-snapshot', 'archive-candidate' or 'author'
    path: str | None = None  # a PDF, or a folder of page pictures
    link: str | None = None  # for a person to download by hand
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**asdict(self), 'want': asdict(self.want)}


def here(w: Want) -> Found | None:
    pdf = next(SCANS.glob(f'IMSLP{w.imslp}-*.pdf'), None)
    if pdf:
        return Found(w, 'here', str(pdf))
    pages = SCANS / 'commons' / f'IMSLP{w.imslp}'
    if pages.exists() and any(pages.glob('*.jpg')):
        return Found(w, 'here', str(pages))
    return None


def _wikimedia(*args: str) -> str:
    run = subprocess.run([os.path.expanduser(t) for t in shlex.split(settings.get('wikimedia'))] + list(args), capture_output=True,
                         text=True, timeout=3600)
    if run.returncode not in (0, 1):
        raise RuntimeError(f'wikimedia.py {args[0]} failed: {run.stderr[-300:]}')
    return run.stdout


def commons(w: Want) -> Found | None:
    """A copy of the very IMSLP file on Commons: titled with its number,
    or with its name, and with the same number of pages."""
    if not settings.get('wikimedia'):
        print('Commons skipped: the wikimedia setting names no Wikimedia client', flush=True)
        return None
    hits = []
    for query in (f'IMSLP{w.imslp}', (w.name or '').rsplit('.', 1)[0].replace('_', ' ')):
        if not query:
            continue
        out = _wikimedia('search', f'{query} filetype:pdf', '--limit', '5', '--json')
        hits += json.loads(out or '[]')
    stem = re.sub(r'[_ ]+', ' ', (w.name or '').rsplit('.', 1)[0]).lower()
    for h in hits:
        title = re.sub(r'[_ ]+', ' ', h['title'][5:].rsplit('.', 1)[0]).lower()
        same = title.startswith(f'imslp{w.imslp}-'.lower()) or (stem and title.endswith(stem))
        if not same:
            continue
        info = json.loads(_wikimedia('info', h['title']) or '[]')
        count = info[0].get('pages') if info else None
        if w.pages and count != w.pages:
            continue
        folder = SCANS / 'commons' / f'IMSLP{w.imslp}'
        saved = _wikimedia('fetch', h['title'], '--out', str(folder)).strip().splitlines()
        if not saved:
            continue
        target = SCANS / f'IMSLP{w.imslp}-{w.name or pathlib.Path(saved[-1]).name}'
        pathlib.Path(saved[-1]).rename(target)
        return Found(w, 'commons', str(target), notes=[f"Commons: {h['title']}"])
    return None


def _same_size(megabytes: float | None, size: int) -> bool:
    return megabytes is not None and any(abs(size / unit - megabytes) <= 0.011 for unit in (1e6, 1 << 20))


def archive_snapshot(w: Want) -> Found | None:
    source = 'http://imslp.org/wiki/' + w.work.replace(' ', '_')
    for item in archive.search(f'collection:imslp AND source:"{source}"'):
        files = {x['name']: x for x in archive.metadata(item['identifier']).get('files', [])}
        if w.name in files and _same_size(w.megabytes, int(files[w.name].get('size', 0))):
            target = SCANS / f'IMSLP{w.imslp}-{w.name}'
            archive.download(item['identifier'], w.name, target)
            return Found(w, 'archive-snapshot', str(target))
    return None


def archive_candidate(w: Want) -> Found | None:
    """A library's scan of the same edition on the Archive: the composer,
    the publisher and a year near the edition's, outside the IMSLP
    snapshot. It is a candidate until pairing confirms it."""
    publisher = re.search(r'(Breitkopf|Peters|Simrock|Schott|Eulenburg|Durand|Hamelle|Augener|Novello|Jurgenson|'
                          r'Belaieff|Hansen|Universal|Kistner|Schlesinger|Litolff|Rahter|Aibl|Hainauer|Ricordi)',
                          w.publisher or '')
    if not publisher:
        return None
    year = re.search(r'1[6-9]\d\d', w.publisher or '')
    surname = w.composer.split(',')[0].replace('_', ' ')
    title = re.sub(r'\(.*?\)|[^A-Za-z0-9 ]', ' ', w.work).strip()
    words = ' '.join(t for t in title.split() if len(t) > 2 and not t.lower().startswith(surname.lower()[:4]))[:80]
    q = (f'mediatype:texts AND creator:({surname}) AND ({words}) AND publisher:({publisher.group(1)}) '
         f'AND NOT collection:imslp')
    if year:
        y = int(year.group(0))
        q += f' AND year:[{y - 3} TO {y + 3}]'
    for item in archive.search(q, fields=('identifier', 'title', 'date', 'publisher'), rows=3):
        files = archive.metadata(item['identifier']).get('files', [])
        pdfs = [f for f in files if f['name'].endswith('.pdf') and not f['name'].endswith('_text.pdf')
                and int(f.get('size', 0)) <= MAX_CANDIDATE_MB * 1e6]
        if not pdfs:
            continue
        f = pdfs[0]
        target = SCANS / 'candidates' / f'IMSLP{w.imslp}' / f"{item['identifier']}.pdf"
        archive.download(item['identifier'], f['name'], target)
        return Found(w, 'archive-candidate', str(target),
                     notes=[f"Archive item {item['identifier']}: {item.get('title')}, {item.get('publisher')}, "
                            f"{item.get('date')}; unconfirmed until it pairs"])
    return None


def mdz_candidate(w: Want) -> Found | None:
    """A public-domain copy at MDZ of a print by the same publisher of the
    same opus, found in the union catalog."""
    opus = re.search(r'(Op\.?\s*\d+)|(K\.?\s*\d+)|(D\.?\s*\d+)|(Hob\.\s*III:?\s*\d+)', w.work)
    publisher = re.search(r'(Breitkopf|Peters|Simrock|Schott|Eulenburg|Durand|Hamelle|Augener|Novello|Jurgenson|'
                          r'Belaieff|Hansen|Universal|Kistner|Schlesinger|Litolff|Rahter|Aibl|Hainauer|Ricordi)',
                          w.publisher or '')
    if not opus or not publisher:
        return None
    number = re.findall(r'\d+', opus.group(0))[0]
    surname = w.composer.split(',')[0].replace('_', ' ')
    for rec in mdz.search(f'marcxml.title="Quartets" and marcxml.title="{number}"', limit=50):
        text = f"{rec['title']} {rec['published']}"
        if publisher.group(1).lower() not in text.lower():
            continue
        folder = SCANS / 'candidates' / f'IMSLP{w.imslp}' / f"mdz-{rec['id']}"
        if mdz.fetch(rec['id'], folder):
            return Found(w, 'mdz-candidate', str(folder),
                         notes=[f"MDZ {rec['id']}: {rec['title'][:80]}, {rec['published'][:60]} "
                                f"({surname}); unconfirmed until it pairs"])
    return None


def author(w: Want) -> Found:
    return Found(w, 'author', link=f'https://imslp.org/wiki/Special:ImagefromIndex/{w.imslp}')


WATERFALL = [here, commons, archive_snapshot, archive_candidate, mdz_candidate]


def find(w: Want) -> Found:
    """The first source that has the scan, or a link for a person. A
    source that fails or refuses is noted and the next one is tried; a
    refusal also halts that source for good (see archive.py)."""
    notes = []
    for step in WATERFALL:
        try:
            got = step(w)
        except Exception as e:
            notes.append(f'{step.__name__}: {e!r}'[:300])
            continue
        if got:
            got.notes = notes + got.notes
            return got
    out = author(w)
    out.notes = notes
    return out


def wanted(lookup: dict, scan: dict | None = None) -> Want:
    """A Want from a row of the catalog lookup, with the file's name and
    size where the scan log has them."""
    scan = scan or {}
    return Want(imslp=lookup['imslp'], name=scan.get('name') or lookup.get('name'),
                megabytes=scan.get('megabytes') or lookup.get('megabytes'), pages=lookup.get('pages'),
                work=lookup['work'], composer=lookup['path'].split('/')[0], publisher=lookup.get('publisher'))
