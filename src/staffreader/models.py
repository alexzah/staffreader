"""The trained models the reader uses by default, published on a GitHub
release and checked against the checksums written here. `staffreader
fetch` downloads them into the cache, where the reader looks for them
(models/<name>/last.pt): the staff reader and the text reader that reads
lyrics and staff labels. They are licensed as the code is."""

from __future__ import annotations

import hashlib
import pathlib

from .render import CACHE

RELEASE = 'models-v1'
URL = f'https://github.com/alexzah/staffreader/releases/download/{RELEASE}'
# Each model's name in the cache, and its file's checksum.
MODELS = {
    'v3-mix': 'b16f71742b7e2a3c27e7dbb9d36700ed4e1efa7b94c2c1762f5a9ba5efd3cf91',
    'text-v4': '0b5ce1b9fcc9c0843910008fa7eed3961522c432a5c05b54c777904b6edb5f38',
}


def fetch() -> list[pathlib.Path]:
    """Download each model not yet in the cache, and check it."""
    import requests
    done = []
    for name, sha256 in MODELS.items():
        out = CACHE / 'models' / name / 'last.pt'
        if out.exists() and _sha256(out) == sha256:
            done.append(out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix('.part')
        with requests.get(f'{URL}/{name}.pt', stream=True, timeout=120) as r:
            r.raise_for_status()
            with part.open('wb') as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        got = _sha256(part)
        if got != sha256:
            part.unlink()
            raise RuntimeError(f'{URL}/{name}.pt has checksum {got}, not {sha256}: not kept')
        part.rename(out)
        done.append(out)
    return done


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
