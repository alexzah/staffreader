"""Pictures of scores we know: the S track.

An excerpt of a score is cut out as MusicXML, carrying what is in force
where it starts, and MuseScore engraves it to PDF in its own home, so the
reader's own settings are never read or changed. Each page is rasterized
in gray at 300 dpi. A scan-like copy adds what printing and scanning do to
a page: a little blur, noise, a slight rotation, uneven paper and ink that
spreads or breaks up when it is thresholded.

The excerpt itself is the answer for these pictures, exactly.
"""

from __future__ import annotations

import copy
import os
import pathlib
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageFilter

from . import canonical, settings


CACHE = settings.path('cache')
MSCORE = settings.get('mscore')
ORDER = ['divisions', 'key', 'time', 'staves', 'clef', 'transpose']


def _close_spanners(part: ET.Element):
    """Take out a slur, hairpin, octave line, pedal mark or dashed line
    that the cut left with one end: its stop before any start, or a start
    that never stops. MuseScore's import, given a stop with no start,
    loses count and drops every later slur of the part."""
    for tags, start_types in ((('slur',), ('start',)),
                              (('wedge',), ('crescendo', 'diminuendo')),
                              (('octave-shift',), ('up', 'down')),
                              (('pedal',), ('start',)),
                              (('dashes', 'bracket'), ('start',))):
        open_: dict[tuple, ET.Element] = {}
        parents = {c: p for p in part.iter() for c in p}
        for el in [e for e in part.iter() if e.tag in tags]:
            kind = el.get('type')
            key = (el.tag, el.get('number', '1'))
            if kind in start_types:
                open_[key] = el
            elif kind == 'stop':
                if key in open_:
                    del open_[key]
                else:
                    parents[el].remove(el)
        for el in open_.values():
            parents[el].remove(el)


def excerpt(source: pathlib.Path, first: int, last: int, out: pathlib.Path) -> pathlib.Path:
    """Measures [first, last] of [source], counted from 1 by position, as
    MusicXML. The first measure carries the divisions, key, time, staves
    and clefs in force where it starts. Directions that span the cut, such
    as hairpins and slurs left open, are left as MusicXML has them; the
    engraving shows them, and the answer compares only what starts inside."""
    root = ET.fromstring(canonical.load(source))
    for part in root.findall('part'):
        measures = part.findall('measure')
        state: dict[str, ET.Element] = {}
        for m in measures[:first]:
            for attributes in m.findall('attributes'):
                for child in attributes:
                    tag = child.tag if child.tag != 'clef' else f"clef{child.get('number', '1')}"
                    state[tag] = child
        keep = measures[first - 1:last]
        for m in measures:
            part.remove(m)
        head = copy.deepcopy(keep[0])
        # The state in force goes where the bar begins: an attributes
        # element further in, such as a clef change after the first notes,
        # is left where it is.
        opening = next((c for c in head if c.tag not in ('print', 'barline')), None)
        attributes = opening if opening is not None and opening.tag == 'attributes' else None
        if attributes is None:
            attributes = ET.Element('attributes')
            head.insert(next((k for k, c in enumerate(head) if c.tag not in ('print', 'barline')), len(head)), attributes)
        have = {c.tag if c.tag != 'clef' else f"clef{c.get('number', '1')}" for c in attributes}
        for tag, el in state.items():
            if tag not in have:
                el = copy.deepcopy(el)
                if tag == 'time':
                    # Carried in from an earlier bar, so not printed here,
                    # as on a page that starts mid-work.
                    el.set('print-object', 'no')
                attributes.append(el)
        kids = sorted(attributes, key=lambda c: next((i for i, t in enumerate(ORDER) if c.tag == t), len(ORDER)))
        for c in list(attributes):
            attributes.remove(c)
        attributes.extend(kids)
        # A pickup cut out of the middle of a score is a full bar here.
        part.append(head)
        part.extend(copy.deepcopy(m) for m in keep[1:])
        _close_spanners(part)
    for credit in root.findall('credit'):
        root.remove(credit)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding='utf-8', xml_declaration=True)
    return out


def mscore_env() -> dict:
    """An environment in which MuseScore runs in its own home, so the
    reader's own settings are never read or changed, and one process's
    settings never meet another's."""
    home = CACHE / 'mscore-home'
    settings = home / f'.config-{os.getpid()}'
    (settings / 'MuseScore').mkdir(parents=True, exist_ok=True)
    (settings / 'MuseScore' / 'MuseScore4.ini').write_text('[import]\nmusicXml\\importInferTextType=false\n')
    return {**os.environ, 'HOME': str(home), 'XDG_CONFIG_HOME': str(settings),
            'XDG_DATA_HOME': str(home / '.local/share'), 'XDG_CACHE_HOME': str(home / '.cache'),
            'QT_QPA_PLATFORM': 'offscreen'}


def engrave(musicxml: pathlib.Path, pdf: pathlib.Path) -> pathlib.Path:
    """[musicxml] engraved by MuseScore to [pdf], or to SVG pages when
    [pdf] ends in .svg: name-1.svg, name-2.svg and on, which mark every
    staff line and barline and so give each page's exact layout."""
    env = mscore_env()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    if pdf.exists():
        pdf.unlink()
    run = subprocess.run([MSCORE, '-o', str(pdf), str(musicxml)], env=env, capture_output=True, text=True,
                         timeout=600)
    made = pdf if pdf.suffix != '.svg' else pdf.with_name(f'{pdf.stem}-1.svg')
    if not made.exists():
        raise RuntimeError(f'MuseScore did not engrave {musicxml}:\n{run.stderr[-2000:]}')
    return pdf


def rasterize(pdf: pathlib.Path, out_dir: pathlib.Path, dpi: int = 300) -> list[pathlib.Path]:
    """Every page of [pdf] as an 8-bit gray PNG, page-01.png and on."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob('page-*.png'):
        old.unlink()
    subprocess.run(['pdftoppm', '-r', str(dpi), '-gray', '-png', str(pdf), str(out_dir / 'page')], check=True)
    pages = sorted(out_dir.glob('page-*.png'))
    # pdftoppm pads the page number to the count's width; make it two digits.
    named = []
    for i, p in enumerate(pages, 1):
        target = out_dir / f'page-{i:02d}.png'
        if p != target:
            p.rename(target)
        named.append(target)
    return named


def scanlike(page: pathlib.Path, out: pathlib.Path, seed: int = 0, strength: float = 1.0) -> pathlib.Path:
    """A copy of [page] as if printed and scanned: uneven paper, a slight
    rotation, blur, noise, and ink thickened or broken by thresholding."""
    rng = np.random.default_rng(seed)
    img = Image.open(page).convert('L')
    angle = float(rng.uniform(-0.6, 0.6)) * strength
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6 + 0.4 * strength))
    a = np.asarray(img, dtype=np.float32) / 255.0
    h, w = a.shape
    # Uneven paper: a smooth, low-frequency tint across the page.
    yy, xx = np.mgrid[0:h, 0:w]
    tint = 0.06 * strength * (np.sin(xx / w * np.pi * rng.uniform(0.5, 2)) * np.cos(yy / h * np.pi * rng.uniform(0.5, 2)))
    paper = 0.93 - tint
    a = a * paper
    a += rng.normal(0, 0.035 * strength, a.shape)
    # Ink thickens a little where the threshold is low, thins where high.
    threshold = 0.55 + rng.uniform(-0.08, 0.08) * strength
    ink = a < threshold
    a = np.where(ink, rng.uniform(0.05, 0.2), np.clip(a + 0.05, 0, 1))
    a = np.clip(a + rng.normal(0, 0.02 * strength, a.shape), 0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=0.4)).save(out)
    return out
