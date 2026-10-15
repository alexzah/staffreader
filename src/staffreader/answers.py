"""The answer keys the ladder cases score against (answers/keys.yaml).

They are scores, so they are never kept in git: they are published as one
zip on a GitHub release, and `staffreader fetch` downloads it into the
cache and checks it against the checksum written here. A maintainer
makes the zip with scripts/build_answer_keys.py, which applies the
corrections each key lists, each read from the printed page, and then
updates RELEASE and SHA256.

A correction names a note by its bar (counted from the first), its onset
in quarter notes from the bar's start and its written pitch, and either
moves it (`to`, with the accidental to print where the key needs one) or
ties it (`tie: start` or `stop`)."""

from __future__ import annotations

import hashlib
import io
import pathlib
import re
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction

import yaml

from .render import CACHE

HERE = pathlib.Path(__file__).resolve().parents[2]
KEYS = HERE / 'answers' / 'keys.yaml'
RELEASE = 'answers-v1'
URL = f'https://github.com/alexzah/staffreader/releases/download/{RELEASE}/{RELEASE}.zip'
SHA256 = '6160cdeb861485fd9ad1bbd7634175cadd3cea5b4dd9829e954aee12bb9aea65'
FOLDER = CACHE / 'answers'


def keys() -> dict:
    return yaml.safe_load(KEYS.read_text())


def path(key: str) -> pathlib.Path:
    """The answer key's file in the cache."""
    found = sorted(FOLDER.glob(f'{key}.*'))
    if not found:
        raise FileNotFoundError(f'no answer key {key} in {FOLDER}: run `staffreader fetch`')
    return found[0]


def fetch() -> pathlib.Path:
    """Download the release's answer keys into the cache and check them."""
    import requests
    if SHA256 is None:
        raise RuntimeError('no release of the answer keys is named yet (answers.SHA256)')
    data = requests.get(URL, timeout=120).content
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        raise RuntimeError(f'{URL} has checksum {got}, not {SHA256}: not unpacked')
    FOLDER.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(FOLDER)
    return FOLDER


def read_musicxml(file: pathlib.Path) -> bytes:
    """A score's MusicXML: the file itself, or the score inside an .mxl."""
    if file.suffix != '.mxl':
        return file.read_bytes()
    with zipfile.ZipFile(file) as z:
        container = ET.fromstring(z.read('META-INF/container.xml'))
        root = container.find('.//{*}rootfile').get('full-path')
        return z.read(root)


def _pitch(text: str) -> tuple[str, int, int]:
    m = re.fullmatch(r'([A-G])(#{1,2}|b{1,2})?(-?\d)', text)
    if not m:
        raise ValueError(f'not a pitch: {text}')
    acc = m.group(2) or ''
    return m.group(1), acc.count('#') - acc.count('b'), int(m.group(3))


def _written(note: ET.Element) -> str | None:
    p = note.find('pitch')
    if p is None:
        return None
    alter = int(float(p.findtext('alter', '0')))
    return p.findtext('step') + ('#' * alter if alter > 0 else 'b' * -alter) + p.findtext('octave')


def _onsets(measure: ET.Element, divisions: int) -> tuple[list[tuple[Fraction, ET.Element]], int]:
    """Each note of a measure with its onset in quarter notes."""
    out, at, last = [], 0, 0
    for el in measure:
        if el.tag == 'attributes' and el.findtext('divisions'):
            divisions = int(el.findtext('divisions'))
        elif el.tag == 'backup':
            at -= int(el.findtext('duration'))
        elif el.tag == 'forward':
            at += int(el.findtext('duration'))
        elif el.tag == 'note':
            if el.find('chord') is not None:
                out.append((Fraction(last, divisions), el))
                continue
            last = at
            out.append((Fraction(at, divisions), el))
            if el.find('grace') is None:
                at += int(el.findtext('duration', '0'))
    return out, divisions


def _set(note: ET.Element, tag: str, el: ET.Element, after: tuple[str, ...]):
    """Put [el] in [note] after the last of [after] it has, as MusicXML orders a note."""
    at = 0
    for k, child in enumerate(list(note)):
        if child.tag in after:
            at = k + 1
    note.insert(at, el)


def correct(musicxml: bytes, corrections: list[dict], part: int = 0) -> bytes:
    """[musicxml] with each correction applied. Each must name exactly one
    note, or nothing is written."""
    root = ET.fromstring(musicxml)
    measures = root.findall('part')[part].findall('measure')
    divisions, by_bar = 1, {}
    for k, m in enumerate(measures, start=1):
        by_bar[k], divisions = _onsets(m, divisions)
    for c in corrections:
        onset = Fraction(str(c['onset']))
        hits = [n for at, n in by_bar[c['bar']] if at == onset and _written(n) == c['pitch']]
        if len(hits) != 1:
            raise ValueError(f'correction {c} names {len(hits)} notes')
        note = hits[0]
        if 'to' in c:
            step, alter, octave = _pitch(c['to'])
            p = note.find('pitch')
            p.find('step').text, p.find('octave').text = step, str(octave)
            if p.find('alter') is not None:
                p.remove(p.find('alter'))
            if alter:
                a = ET.Element('alter')
                a.text = str(alter)
                p.insert(1, a)
            if note.find('accidental') is not None:
                note.remove(note.find('accidental'))
            if c.get('accidental'):
                a = ET.Element('accidental')
                a.text = c['accidental']
                _set(note, 'accidental', a, ('pitch', 'unpitched', 'rest', 'duration', 'tie', 'voice', 'type', 'dot'))
        if 'tie' in c:
            _set(note, 'tie', ET.Element('tie', type=c['tie']), ('pitch', 'unpitched', 'rest', 'duration', 'tie'))
            notations = note.find('notations')
            if notations is None:
                notations = ET.Element('notations')
                _set(note, 'notations', notations, ('pitch', 'duration', 'tie', 'voice', 'type', 'dot', 'accidental',
                                                    'time-modification', 'stem', 'notehead', 'staff', 'beam'))
            notations.append(ET.Element('tied', type=c['tie']))
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)
