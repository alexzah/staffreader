"""Reading IMSLP's catalog within IMSLP's rules, and nothing more.

IMSLP's robots.txt forbids automated access to /images/ and /imglnks/,
where every PDF is served, to Special: pages (Special:ReverseLookup among
them), to File: pages and to /index.php. It allows api.php and asks for
two seconds between requests. So this module:

- reads only through api.php;
- waits at least GAP seconds between requests, across every process on
  the machine, through a lock file, so two tools running at once never
  double the rate;
- sends a contact User-Agent, and maxlag so a busy server can say no;
- stops at the first refusal or odd answer, and leaves a note in
  STOP_FILE that halts every later request until a person removes it;
- never downloads a score. For a PDF it gives the link a person opens
  in a browser to download it by hand."""

from __future__ import annotations

import fcntl
import html
import json
import re
import subprocess
import time
import urllib.parse
from dataclasses import asdict, dataclass

from . import contact
from .render import CACHE

API = 'https://imslp.org/api.php'
GAP = 5.0
LOCK = CACHE / 'imslp.lock'
STOP_FILE = CACHE / 'imslp.STOPPED'


class Refused(RuntimeError):
    """IMSLP declined or answered oddly; nothing more is asked of it."""


def _wait_turn():
    """Block until GAP seconds have passed since any process's last request."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        last = float(f.read().strip() or 0)
        time.sleep(max(0.0, last + GAP - time.time()))
        f.seek(0)
        f.truncate()
        f.write(str(time.time()))


def api(**params) -> dict:
    """One api.php request, in turn, or Refused."""
    if STOP_FILE.exists():
        raise Refused(f'IMSLP access is halted: {STOP_FILE.read_text().strip()}')
    _wait_turn()
    args = ['curl', '-s', '--compressed', '--max-time', '60', '-A', contact.user_agent(short=True), '-w', '\n%{http_code}', '-G', API]
    for k, v in {**params, 'format': 'json', 'maxlag': '5'}.items():
        args += ['--data-urlencode', f'{k}={v}']
    out = subprocess.run(args, capture_output=True, text=True).stdout
    body, _, status = out.rpartition('\n')
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None
    if status != '200' or data is None or data.get('error', {}).get('code') in ('maxlag', 'ratelimited'):
        note = f'{time.strftime("%Y-%m-%d %H:%M")} HTTP {status}: {body[:200]!r}'
        STOP_FILE.write_text(note + '\n')
        raise Refused(note)
    return data


@dataclass
class Page:
    title: str
    html: str
    text: str


def page(title: str) -> Page | None:
    """A work page, as IMSLP renders it and as plain text, or None if
    there is none."""
    d = api(action='parse', page=title, prop='text', redirects='1')
    if 'error' in d:
        return None
    raw = re.sub(r'<script.*?</script>', '', d['parse']['text']['*'], flags=re.S)
    return Page(d['parse']['title'], raw, re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', raw))))


def search(query: str, limit: int = 5) -> list[str]:
    d = api(action='query', list='search', srsearch=query, srnamespace='0', srlimit=str(limit))
    return [h['title'] for h in d.get('query', {}).get('search', [])]


@dataclass
class File:
    """One file on a work page, as the page describes it."""
    id: str
    work: str
    name: str | None = None  # the file's name on IMSLP, as File:<name>
    megabytes: float | None = None
    description: str | None = None
    pages: int | None = None
    publisher: str | None = None
    copyright: str | None = None

    @property
    def link(self) -> str:
        """Where a person downloads it, in a browser."""
        return f'https://imslp.org/wiki/Special:ImagefromIndex/{self.id}'

    def to_dict(self) -> dict:
        return {**asdict(self), 'link': self.link}


def file_on(pg: Page, fid: str) -> File | None:
    """File #[fid] as its work page describes it."""
    text = pg.text
    at = text.find(f'#{fid} -')
    if at < 0:
        return None
    out = File(fid, pg.title)
    link = re.search(r'href="/wiki/File:([^"]+)"[^>]*>#' + re.escape(fid) + r'</a>\s*-\s*([\d.]+)MB', pg.html)
    if link:
        out.name = urllib.parse.unquote(html.unescape(link.group(1)))
        out.megabytes = float(link.group(2))
    size = re.match(r'#\S+ - ([\d.]+MB), (\d+) pp', text[at:])
    out.pages = int(size.group(2)) if size else None
    pub = re.search(r'Pub lisher \. Info\.(.*?)(Copyright|Misc\. Notes|Purchase)', text[at:at + 3000])
    out.publisher = re.sub(r'\s+([,.])', r'\1', pub.group(1)).strip()[:300] if pub else None
    cr = re.search(r'Copyright (.*?) \[', text[at:at + 3000])
    out.copyright = cr.group(1).strip()[:80] if cr else None
    out.description = text[max(0, at - 120):at].split('Purchase Javascript is required for this feature.')[-1] \
        .strip()[-100:] or None
    return out


def files_on(pg: Page) -> list[File]:
    """Every score a work page lists (its PDFs, not its recordings), as
    the page describes each."""
    ids = dict.fromkeys(re.findall(r'href="/wiki/File:[^"]+\.pdf"[^>]*>#(\d+)</a>', pg.html, re.IGNORECASE))
    return [f for fid in ids if (f := file_on(pg, fid))]


def find_file(fid: str, titles: list[str], query: str | None = None) -> File | None:
    """File #[fid], looked for on the work pages [titles], then on what a
    search for [query] finds. IMSLP's own lookup by number is a Special:
    page, which robots.txt forbids, so the work must be guessed."""
    fid = fid.strip().lstrip('#')
    tried = set()
    for title in titles:
        pg = page(title)
        tried.add(title)
        if pg and (f := file_on(pg, fid)):
            return f
    if query:
        for title in search(query, 4)[:3]:
            if title in tried:
                continue
            pg = page(title)
            if pg and (f := file_on(pg, fid)):
                return f
    return None
