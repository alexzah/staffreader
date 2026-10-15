"""What a test needs beyond the code, so a machine without it skips the
test with the reason rather than failing: the answer keys (staffreader
fetch), MuseScore 4 (the mscore setting), and a case's printed scan."""

from __future__ import annotations

import pathlib
import shutil

import pytest

from staffreader import answers, cases, render

HAVE_KEYS = answers.FOLDER.exists()
HAVE_MUSESCORE = shutil.which(render.MSCORE) is not None or pathlib.Path(render.MSCORE).expanduser().is_file()

keys = pytest.mark.skipif(not HAVE_KEYS, reason='the answer keys are not fetched (staffreader fetch)')
musescore = pytest.mark.skipif(not HAVE_MUSESCORE, reason='MuseScore 4 is not installed (the mscore setting)')


def scan(track) -> None:
    """Skip a test of a scan track whose scan is not on this machine."""
    if track.kind == 'scan' and not (cases.SCANS / track.pdf).exists():
        pytest.skip(f'the scan {track.pdf} is not in the scans folder')
