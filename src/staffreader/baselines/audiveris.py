"""Audiveris 5.11, run as a printed page has always been read for the
score pipeline: the page is rendered so its staff lines lie about
TARGET_INTERLINE pixels apart (measured from a first rendering at
PROBE_DPI), read in batch, and exported.

Audiveris is the Flathub build (org.audiveris.audiveris), given access to
the work folder for each run only, unless the audiveris setting names
another command (settings.py). It reads text with Tesseract in Italian, the language of
most printed tempo and playing directions."""

from __future__ import annotations

import os
import pathlib
import re
import shlex
import subprocess
import zipfile
from .. import settings

NAME = 'audiveris'
FLATPAK = 'org.audiveris.audiveris'
TARGET_INTERLINE = 20
PROBE_DPI = 300
OCR_LANGUAGE = 'ita'


def _audiveris(work: pathlib.Path, *args: str) -> str:
    """Runs Audiveris in batch and returns its log."""
    work = work.resolve()
    own = settings.get('audiveris')
    command = [os.path.expanduser(t) for t in shlex.split(own)] if own else ['flatpak', 'run', f'--filesystem={work}', FLATPAK]
    run = subprocess.run(command + ['-batch', '-constant',
                                    f'org.audiveris.omr.text.Language.defaultSpecification={OCR_LANGUAGE}', *args],
                         capture_output=True, text=True)
    return run.stdout + run.stderr


def _render(pdf: pathlib.Path, page: int, dpi: int, out: pathlib.Path) -> pathlib.Path:
    """The page as a gray PNG at [dpi]. Audiveris binarizes it itself;
    rendered in black and white at a low resolution, the thin staff lines
    break up."""
    subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-r', str(dpi), '-gray',
                    '-singlefile', '-png', str(pdf), str(out.with_suffix(''))], check=True)
    return out


def _interline(log: str) -> int:
    match = re.search(r'interline\((\d+),(\d+),(\d+)\)', log)
    if not match:
        raise RuntimeError('Audiveris found no staff lines:\n' + log[-2000:])
    return int(match.group(2))


def _read(pdf: pathlib.Path, page: int, work: pathlib.Path) -> pathlib.Path:
    out = work / f'page-{page:02d}'
    out.mkdir(parents=True, exist_ok=True)
    probe = _render(pdf, page, PROBE_DPI, out / 'probe.png')
    interline = _interline(_audiveris(work, '-step', 'SCALE', '-output', str((out / 'probe').resolve()),
                                      '--', str(probe.resolve())))
    dpi = round(PROBE_DPI * TARGET_INTERLINE / interline / 10) * 10
    image = _render(pdf, page, dpi, out / 'page.png')
    log = _audiveris(work, '-export', '-save', '-output', str((out / 'audiveris').resolve()),
                     '--', str(image.resolve()))
    (out / 'audiveris.log').write_text(log)
    return out


def read(image: pathlib.Path, page: int | None, work: pathlib.Path) -> pathlib.Path:
    """The MusicXML Audiveris reads from one page: a page of a PDF, or a
    PNG, which is wrapped losslessly as a one-page PDF at 300 dpi."""
    work.mkdir(parents=True, exist_ok=True)
    if page is None:
        pdf = work / 'input.pdf'
        subprocess.run(['magick', str(image), '-units', 'PixelsPerInch', '-density', '300',
                        '-compress', 'lossless', str(pdf)], check=True)
        image, page = pdf, 1
    try:
        _read(pathlib.Path(image), page, work)
    except RuntimeError:
        pass  # no staff lines: reported below as no MusicXML
    found = sorted((work / f'page-{page:02d}' / 'audiveris').rglob('*.mxl'))
    if not found:
        log = work / f'page-{page:02d}' / 'audiveris.log'
        text = log.read_text(errors='ignore') if log.exists() else ''
        cause = 'it ran out of Java memory' if 'OutOfMemoryError' in text else 'see its log'
        raise RuntimeError(f'Audiveris wrote no MusicXML: {cause} ({log})')
    out = work / 'reading.musicxml'
    with zipfile.ZipFile(found[0]) as z:
        name = next(n for n in z.namelist() if n.endswith(('.xml', '.musicxml')) and not n.startswith('META'))
        out.write_bytes(z.read(name))
    return out
