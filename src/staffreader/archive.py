"""Fetching scans from the Internet Archive within its rules.

The Archive's guide for bots and AI agents (archive.org/developers/
bots.html, read 2026-10-12) asks automated tools to:

- send a User-Agent naming the tool, its version and, for an AI agent,
  the model;
- wait between requests in bulk work, and honor 429 Too Many Requests
  and Retry-After;
- not download unchanged files again (checksums);
- cache responses, and limit concurrency;
- back off exponentially on errors (a server error is retried, then
  given up for that file alone).

Its robots.txt disallows only /control/ and /report/. So this module
makes one request at a time across every process on the machine (a lock
file), at least GAP seconds apart; caches every metadata answer; checks
each download's MD5 against the item's metadata and skips files already
here; waits as Retry-After says on a 429 or 503, doubling each time; and
stops for good, leaving STOP_FILE, after a refusal it cannot wait out."""

from __future__ import annotations

import fcntl
import hashlib
import json
import pathlib
import time

import requests

from . import contact
from .render import CACHE

GAP = 2.0
LOCK = CACHE / 'archive.lock'
STOP_FILE = CACHE / 'archive.STOPPED'
META = CACHE / 'archive-meta'


class Refused(RuntimeError):
    """The Archive declined; nothing more is asked of it."""


def _wait_turn():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        last = float(f.read().strip() or 0)
        time.sleep(max(0.0, last + GAP - time.time()))
        f.seek(0)
        f.truncate()
        f.write(str(time.time()))


def _get(url: str, *, params: dict | None = None, stream: bool = False) -> requests.Response:
    if STOP_FILE.exists():
        raise Refused(f'Archive access is halted: {STOP_FILE.read_text().strip()}')
    wait = 5.0
    for _ in range(5):
        _wait_turn()
        r = requests.get(url, params=params, headers={'User-Agent': contact.user_agent()}, stream=stream, timeout=120)
        if r.status_code in (429, 500, 502, 503, 504):
            after = r.headers.get('Retry-After', '')
            time.sleep(max(wait, float(after)) if after.isdigit() else wait)
            wait *= 2
            continue
        if r.status_code == 200:
            return r
        break
    note = f'{time.strftime("%Y-%m-%d %H:%M")} HTTP {r.status_code} for {url}'
    if r.status_code in (403, 429, 503):
        STOP_FILE.write_text(note + '\n')
    raise Refused(note)


def metadata(identifier: str) -> dict:
    """An item's metadata and file list, cached."""
    META.mkdir(parents=True, exist_ok=True)
    cached = META / f'{identifier}.json'
    if cached.exists():
        return json.loads(cached.read_text())
    data = _get(f'https://archive.org/metadata/{identifier}').json()
    cached.write_text(json.dumps(data))
    return data


def search(query: str, fields: tuple[str, ...] = ('identifier', 'title', 'source'), rows: int = 20) -> list[dict]:
    """Items matching an Archive search, cached."""
    META.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f'{query}|{fields}|{rows}'.encode()).hexdigest()[:16]
    cached = META / f'search-{key}.json'
    if cached.exists():
        return json.loads(cached.read_text())
    params = {'q': query, 'fl[]': list(fields), 'rows': rows, 'output': 'json'}
    docs = _get('https://archive.org/advancedsearch.php', params=params).json()['response']['docs']
    cached.write_text(json.dumps(docs))
    return docs


def download(identifier: str, name: str, out: pathlib.Path) -> pathlib.Path:
    """One file of an item, checked against the MD5 its metadata gives.
    A file already here with the right checksum is not fetched again."""
    files = {f['name']: f for f in metadata(identifier).get('files', [])}
    want = files[name].get('md5')
    if out.exists() and want and hashlib.md5(out.read_bytes()).hexdigest() == want:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + '.part')
    r = _get(f'https://archive.org/download/{identifier}/{requests.utils.quote(name)}', stream=True)
    digest = hashlib.md5()
    with open(part, 'wb') as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
            digest.update(chunk)
    if want and digest.hexdigest() != want:
        part.unlink()
        raise RuntimeError(f'{identifier}/{name}: checksum differs from the item metadata')
    part.rename(out)
    return out
