"""Scans from the Bavarian State Library's digital collections (MDZ),
within its rules.

MDZ's interfaces page (digitale-sammlungen.de/en/interfaces, read
2026-10-12) offers the IIIF Presentation API, unlimited, for an item's
manifest, and the IIIF Image API, limited per day, for its pages. Whole
PDFs and high-resolution orders need a person to accept the conditions
of use or solve a CAPTCHA, so they are never scripted. Its robots.txt
allows everything. Each item carries its own rights statement: only
items under the Public Domain Mark are taken, since 'No copyright -
non-commercial use only' (Google's scans) would not suit free training data.
MDZ asks that a picture be credited with the holding library, the call
number, the page and the URN, so every page is saved with a record of
them.

Items are found through the Bavarian union catalog's public SRU search
(bvbr.bib-bvb.de), whose records link to their digital copies.

Requests are one at a time, three seconds apart; page images count
against DAILY_PAGES, a budget kept well inside MDZ's daily limit; and a
refusal halts MDZ for good, leaving STOP_FILE."""

from __future__ import annotations

import datetime
import fcntl
import json
import pathlib
import re
import time

import requests

from . import contact
from .render import CACHE

GAP = 3.0
DAILY_PAGES = 400
LOCK = CACHE / 'mdz.lock'
STOP_FILE = CACHE / 'mdz.STOPPED'
BUDGET = CACHE / 'mdz-budget.json'
SRU = 'http://bvbr.bib-bvb.de:5661/bvb01sru'
MANIFEST = 'https://api.digitale-sammlungen.de/iiif/presentation/v2/{id}/manifest'
PUBLIC_DOMAIN = ('publicdomain/mark', 'creativecommons.org/publicdomain/zero')


class Refused(RuntimeError):
    pass


def _get(url: str, params: dict | None = None) -> requests.Response:
    if STOP_FILE.exists():
        raise Refused(f'MDZ access is halted: {STOP_FILE.read_text().strip()}')
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        last = float(f.read().strip() or 0)
        time.sleep(max(0.0, last + GAP - time.time()))
        r = requests.get(url, params=params, headers={'User-Agent': contact.user_agent()}, timeout=120)
        f.seek(0)
        f.truncate()
        f.write(str(time.time()))
    if r.status_code in (403, 429, 503):
        STOP_FILE.write_text(f'{time.strftime("%Y-%m-%d %H:%M")} HTTP {r.status_code} for {url}\n')
        raise Refused(f'HTTP {r.status_code}')
    r.raise_for_status()
    return r


def _spend(pages: int) -> bool:
    """Take [pages] from today's budget, or refuse if it would overrun."""
    today = datetime.date.today().isoformat()
    spent = json.loads(BUDGET.read_text()) if BUDGET.exists() else {}
    if spent.get(today, 0) + pages > DAILY_PAGES:
        return False
    BUDGET.write_text(json.dumps({today: spent.get(today, 0) + pages}))
    return True


def search(query: str, limit: int = 20) -> list[dict]:
    """Catalog records with an MDZ copy: title, publication and the MDZ
    object id."""
    params = {'version': '1.1', 'operation': 'searchRetrieve', 'recordSchema': 'marcxml',
              'maximumRecords': str(limit), 'query': query}
    t = _get(SRU, params).text
    out = []
    for rec in re.findall(r'<record xmlns.*?</record>', t, re.S):
        ids = re.findall(r'bsb(\d{8})', rec)
        if not ids:
            continue

        def field(tag):
            f = re.findall(rf'<datafield tag="{tag}"[^>]*>(.*?)</datafield>', rec, re.S)
            return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', f[0])).strip() if f else ''
        out.append({'id': f'bsb{ids[0]}', 'title': field('245'), 'published': field('264') or field('260')})
    return out


def manifest(object_id: str) -> dict:
    cached = CACHE / 'mdz-meta' / f'{object_id}.json'
    if cached.exists():
        return json.loads(cached.read_text())
    data = _get(MANIFEST.format(id=object_id)).json()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(data))
    return data


def public_domain(m: dict) -> bool:
    rights = json.dumps(m.get('license', '')) + json.dumps(m.get('attribution', ''))
    return any(p in rights for p in PUBLIC_DOMAIN)


def fetch(object_id: str, out: pathlib.Path) -> pathlib.Path | None:
    """Every page of a public-domain item, as MDZ stores it, with its
    credit beside each. None if the item is not in the public domain or
    today's budget cannot take it."""
    m = manifest(object_id)
    if not public_domain(m):
        return None
    canvases = m['sequences'][0]['canvases']
    todo = [(k, c) for k, c in enumerate(canvases, start=1) if not (out / f'page-{k:04d}.jpg').exists()]
    if not _spend(len(todo)):
        return None
    out.mkdir(parents=True, exist_ok=True)
    meta = {d.get('label'): d.get('value') for d in m.get('metadata', []) if isinstance(d, dict)}
    for k, c in todo:
        service = c['images'][0]['resource']['service']['@id']
        (out / f'page-{k:04d}.jpg').write_bytes(_get(f'{service}/full/full/0/default.jpg').content)
        (out / f'page-{k:04d}.jpg.json').write_text(json.dumps({
            'library': 'Bayerische Staatsbibliothek München', 'object': object_id, 'page': k,
            'label': c.get('label'), 'license': m.get('license'), 'metadata': meta,
        }, ensure_ascii=False))
    return out
