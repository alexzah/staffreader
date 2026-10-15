"""oemer (MIT), an open end-to-end OMR built on segmentation networks,
run from a virtual environment of its own under the cache, so its
dependencies never meet ours."""

from __future__ import annotations

import os
import pathlib
import subprocess

from .. import render

VENV = render.CACHE / 'venvs' / 'oemer'
NAME = 'oemer'


def read(image: pathlib.Path, page: int | None, work: pathlib.Path) -> pathlib.Path:
    """The MusicXML oemer reads from one page image. A PDF page is
    rendered in gray at 300 dpi first."""
    work.mkdir(parents=True, exist_ok=True)
    if page is not None:
        png = work / 'input.png'
        subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-r', '300', '-gray', '-singlefile',
                        '-png', str(image), str(png.with_suffix(''))], check=True)
        image = png
    out_dir = work / 'oemer'
    out_dir.mkdir(exist_ok=True)
    env = {**os.environ, 'PATH': f"{VENV / 'bin'}:{os.environ['PATH']}"}
    run = subprocess.run([str(VENV / 'bin' / 'oemer'), str(image), '-o', str(out_dir)],
                         capture_output=True, text=True, env=env, timeout=3600)
    found = sorted(out_dir.glob('*.musicxml'))
    if not found:
        last = [l for l in run.stderr.splitlines() if l.strip()][-1:] or ['no output']
        (work / 'oemer.log').write_text(run.stdout + run.stderr)
        raise RuntimeError(f'oemer wrote no MusicXML: it stopped with {last[0][:120]} ({work / "oemer.log"})')
    return found[0]
