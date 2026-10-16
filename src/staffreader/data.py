"""Training data: staves engraved by MuseScore from PDMX scores, each with
the tokens it prints.

`select` picks the PDMX scores released as CC0 or public domain, with no
license conflict, one copy of each, that hold a part on one staff whose
bars the page can show. It leaves out every score whose melody matches a
work that must never train (corpus_grams): every answer key's, so the
reader is never trained on a work it is tested on. `render` has MuseScore engrave each chosen score in a music font and
page shape picked at random, and cuts every system's staff of each such part
out of the page, with the tokens of the bars that system holds.

PDMX is read from the folder the pdmx setting names (settings.py).
Everything made here lives in CACHE/data."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import random
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from . import canonical, cases, staffcrop, tokens
from . import render as render_mod
from .render import CACHE
from . import settings

PDMX = settings.path('pdmx')
# A corpus of score bundles whose works never train, beside the answer keys.
BUNDLES = settings.path('scores') / 'bundles' if settings.get('scores') else None
DATA = CACHE / 'data'
# MuseScore's music fonts, weighted toward those that look engraved.
# Leland, MuseScore's default, engraves the synthetic test pages, so the
# reader never sees it in training.
FONTS = {'Bravura': 0.22, 'Emmentaler': 0.22, 'Finale Maestro': 0.2, 'Gonville': 0.16,
         'Finale Broadway': 0.08, 'Petaluma': 0.06, 'MuseJazz': 0.06}
GRAM = 12
STEP_NUMBER = {s: i for i, s in enumerate('CDEFGAB')}


# ---------------------------------------------------------------- melodies

def _melody(bars: list[list[str]]) -> list[tuple[int, str]]:
    """The staff's notes as diatonic steps with their types, chord notes
    left out, so a transposed copy of a tune keeps its intervals and
    rhythm."""
    out, chord = [], False
    for bar in bars:
        for t in bar:
            if t == '+':
                chord = True
            elif t.startswith('N:'):
                if not chord:
                    out.append([7 * int(t[3:]) + STEP_NUMBER[t[2]], ''])
                chord = False
            elif t.startswith('d:') and out and not out[-1][1]:
                out[-1][1] = t[2:]
    return [tuple(n) for n in out]


def _grams(melody: list[tuple[int, str]]) -> set[int]:
    """Runs of GRAM steps, each an interval and how the note's length
    compares with the one before, so a copy in halved values matches."""
    rank = {t: i for i, t in enumerate(tokens.TYPES)}
    steps = [(b[0] - a[0], rank.get(b[1], 3) - rank.get(a[1], 3)) for a, b in zip(melody, melody[1:])]
    grams = set()
    for i in range(len(steps) - GRAM + 1):
        g = tuple(steps[i:i + GRAM])
        if len({x[0] for x in g}) > 2:  # a scale or a repeated figure says nothing of the work
            grams.add(hash(g))
    return grams


def _parts(root: ET.Element):
    for part in root.findall('part'):
        count = max([int(s.text) for s in part.iter('staves') if s.text and s.text.isdigit()] or [1])
        yield part, count


def _root(xml: bytes) -> ET.Element:
    root = ET.fromstring(xml)
    for el in root.iter():
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]
    return root


def corpus_grams() -> set[int]:
    """The melodic fingerprint of every work that must never train: each
    answer key's, and each score in a corpus of bundles where the scores
    setting names one."""
    from . import answers
    files = sorted(answers.FOLDER.glob('*.musicxml'))
    if BUNDLES is not None and BUNDLES.exists():
        files += [d / 'score.musicxml' for d in BUNDLES.iterdir()]
        files += list((BUNDLES.parent / 'work').glob('*/movement.musicxml'))
        files += list((BUNDLES.parent / 'align-agent' / 'work').glob('*/movement.musicxml'))
    for f in files:
        if not f.exists():
            continue
        for part, count in _parts(_root(canonical.load(f))):
            for s in range(1, count + 1):
                grams |= _grams(_melody(tokens.from_part(part, s)[0]))
    return grams


# --------------------------------------------------------------- selection

def _pdmx_rows():
    csv.field_size_limit(10 ** 9)
    with open(PDMX / 'PDMX.csv') as f:
        for r in csv.DictReader(f):
            if r['subset:no_license_conflict'] == 'True' and r['subset:deduplicated'] == 'True' \
                    and r['license'] in ('publicdomain', 'cc-zero'):
                yield r


_CORPUS: set[int] = set()


def _init_select(grams):
    global _CORPUS
    _CORPUS = grams


def _survey(row: dict) -> dict | None:
    """What a PDMX score offers: its parts whose staves the page shows,
    bar by bar, and how much of its melody the works that never train share."""
    try:
        root = _root(canonical.load(PDMX / row['mxl']))
    except Exception:
        return None
    parts, grams, shared = [], set(), 0
    staves = []
    for k, (part, count) in enumerate(_parts(root)):
        staves.append(count)
        shares = []
        for st in range(1, count + 1):
            try:
                bars, states = tokens.from_part(part, st)
            except Exception:
                return None
            if not bars:
                continue
            shares.append(sum(s['readable'] for s in states) / len(states))
            grams |= _grams(_melody(bars))
        if shares and min(shares) >= 0.8:
            parts.append(k)
    if not parts:
        return None
    shared = len(grams & _CORPUS) / len(grams) if grams else 0.0
    return {'mxl': row['mxl'], 'title': row['title'], 'parts': parts, 'staves': staves,
            'shared': round(shared, 3), 'shared_grams': len(grams & _CORPUS)}


def shares_a_work(survey: dict) -> bool:
    """Whether a score shares enough melody with the corpus to be one of
    its works, or an arrangement of one. Leaving out a stranger costs a
    little training data; letting a test work in would void the test."""
    return survey['shared_grams'] >= 15 or (survey['shared'] >= 0.1 and survey['shared_grams'] >= 5)


def select(workers: int = 16, limit: int | None = None, name: str = 'scores') -> pathlib.Path:
    """Survey PDMX and write the scores fit for training, and those left
    out for sharing a melody with the corpus."""
    DATA.mkdir(parents=True, exist_ok=True)
    grams = corpus_grams()
    rows = list(_pdmx_rows())
    if limit:
        random.Random(0).shuffle(rows)
        rows = rows[:limit]
    from .progress import Progress
    bar = Progress(f'select {name}', len(rows), 'scores', 'surveying PDMX for staves the page shows')
    kept, excluded = [], []
    with ProcessPoolExecutor(workers, initializer=_init_select, initargs=(grams,)) as pool:
        for k, got in enumerate(pool.map(_survey, rows, chunksize=64), start=1):
            if k % 2000 == 0:
                bar.update(k, f'{len(kept)} kept, {len(excluded)} left out as works that never train')
            if got is None:
                continue
            (excluded if shares_a_work(got) else kept).append(got)
    bar.finish(f'{len(kept)} kept, {len(excluded)} left out as works that never train')
    out = DATA / f'{name}.jsonl'
    out.write_text(''.join(json.dumps(k) + '\n' for k in kept))
    (DATA / f'{name}-excluded.jsonl').write_text(''.join(json.dumps(k) + '\n' for k in excluded))
    return out


# --------------------------------------------------------------- rendering

VOICE_CLEFS = [(re.compile(r'sopran|canto|discant|cantus|superius', re.I), '1'),
               (re.compile(r'\balt\b|alto|contralto|altus', re.I), '3'),
               (re.compile(r'tenor', re.I), '4')]


def _clean(xml: bytes, tenor: bool = False, quiet_tuplets: bool = False, vocal: bool = False,
           names: random.Random | None = None) -> str:
    """The score as MuseScore should engrave it for training: every bar
    drawn, nothing folded into a multi-bar rest, and laid out by
    MuseScore's own style rather than the upload's. With [tenor], each
    one-staff part in bass clef whose notes lie mostly in the tenor
    range is written in tenor clef instead, as cello, bassoon and
    trombone parts are, since few uploads use it. With [quiet_tuplets],
    tuplets are drawn without their number or bracket, as many printed
    editions draw triplets once the first group has shown them; the bars
    still say they are triplets, so the labels do too. With [vocal], each
    one-staff voice part in a treble clef is written in the C clef older
    editions print it in: soprano, alto or tenor clef. With [names] (a
    random source), each part an older edition would name gets such a
    name, full and short (instruments.historical), and the score counts
    them in its 'renamed' attribute."""
    root = _root(xml)
    if quiet_tuplets:
        for t in root.iter('tuplet'):
            t.set('show-number', 'none')
            t.set('bracket', 'no')
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ('measure-style', 'print', 'defaults'):
                parent.remove(child)
    if tenor:
        for part in root.findall('part'):
            if any(a.findtext('staves') not in (None, '1') for a in part.iter('attributes')):
                continue
            clefs = list(part.iter('clef'))
            if not clefs or any((c.findtext('sign'), c.findtext('line'), c.findtext('clef-octave-change'))
                                != ('F', '4', None) for c in clefs):
                continue
            steps = 'CDEFGAB'
            heights = sorted(7 * int(p.findtext('octave')) + steps.index(p.findtext('step'))
                             for p in part.iter('pitch') if p.findtext('octave') and p.findtext('step') in steps)
            # The median note at D3 or higher (7 * 3 + 1), where a part
            # reads on fewer ledger lines in tenor clef.
            if len(heights) < 16 or heights[len(heights) // 2] < 7 * 3 + 1:
                continue
            for c in clefs:
                c.find('sign').text = 'C'
                c.find('line').text = '4'
            part.set('tenor', 'yes')
    if vocal:
        names = {sp.get('id'): sp.findtext('part-name') or '' for sp in root.iter('score-part')}
        for part in root.findall('part'):
            if any(a.findtext('staves') not in (None, '1') for a in part.iter('attributes')):
                continue
            line = next((n for rx, n in VOICE_CLEFS if rx.search(names.get(part.get('id'), ''))), None)
            clefs = list(part.iter('clef'))
            if line is None or not clefs or any(c.findtext('sign') != 'G' for c in clefs):
                continue
            for c in clefs:  # the pitches are written as they sound, whatever the clef
                c.find('sign').text = 'C'
                c.find('line').text = line
                if (o := c.find('clef-octave-change')) is not None:
                    c.remove(o)
            part.set('vocal', 'yes')
    if names is not None:
        from . import instruments
        renamed = 0
        for sp in root.iter('score-part'):
            got = instruments.historical(sp.findtext('part-name') or sp.findtext('part-abbreviation'), names)
            if got is None:
                continue
            for tag in ('part-name-display', 'part-abbreviation-display'):
                for el in sp.findall(tag):
                    sp.remove(el)
            for tag, text in zip(('part-name', 'part-abbreviation'), got):
                el = sp.find(tag)
                if el is None:
                    el = ET.SubElement(sp, tag)
                el.text = text
                el.attrib.pop('print-object', None)
            renamed += 1
        root.set('renamed', str(renamed))
    return ET.tostring(root, encoding='unicode')


NAME_FACES = ['Liberation Serif', 'DejaVu Serif', 'Libre Baskerville', 'IBM Plex Serif', 'Edwin']


def style(rng: random.Random, names: bool = False, multirest: bool = False) -> tuple[str, str]:
    """A MuseScore style picked at random: a music font, a staff size, a
    page shape and spacing. Returns the font and the style file's text.
    With [names], staff labels are set in a serif face chosen at random,
    bold as often as not, as older editions set them."""
    font = rng.choices(list(FONTS), weights=list(FONTS.values()))[0]
    width = rng.uniform(7.5, 9.0)
    margin = rng.uniform(0.4, 0.8)
    values = {
        'musicalSymbolFont': font, 'musicalTextFont': f'{font} Text',
        'spatium': rng.uniform(1.4, 2.0), 'pageWidth': width, 'pageHeight': rng.uniform(11.0, 12.5),
        'pagePrintableWidth': width - 2 * margin, 'pageEvenLeftMargin': margin, 'pageOddLeftMargin': margin,
        'staffLineWidth': rng.uniform(0.08, 0.13), 'stemWidth': rng.uniform(0.09, 0.13),
        'barWidth': rng.uniform(0.15, 0.22), 'minNoteDistance': rng.uniform(0.3, 0.6),
        'measureSpacing': rng.uniform(1.0, 1.5), 'staffDistance': rng.uniform(5.0, 8.0),
        'minSystemDistance': rng.uniform(7.0, 10.0), 'showMeasureNumber': rng.choice([0, 1, 1]),
        'createMultiMeasureRests': int(multirest), 'hideEmptyStaves': 0,
    }
    if multirest:
        values['minEmptyMeasures'] = rng.choice([2, 2, 3])
    if names:
        face, bold = rng.choice(NAME_FACES), rng.choice([0, 1])
        values.update({'longInstrumentFontFace': face, 'shortInstrumentFontFace': face,
                       'longInstrumentFontStyle': bold, 'shortInstrumentFontStyle': bold,
                       'longInstrumentFontSize': rng.uniform(9.0, 12.0), 'shortInstrumentFontSize': rng.uniform(9.0, 12.0)})
    body = ''.join(f'    <{k}>{round(v, 3) if isinstance(v, float) else v}</{k}>\n' for k, v in values.items())
    return font, f'<?xml version="1.0" encoding="UTF-8"?>\n<museScore version="4.70">\n  <Style>\n{body}  </Style>\n</museScore>\n'


def staves_of(svg: pathlib.Path) -> tuple[list[dict], float, float]:
    """The staves MuseScore drew on a page, top to bottom, in SVG units,
    with the page's width and the staff space."""
    root = ET.parse(svg).getroot()
    lines = []
    for el in root.iter():
        if el.get('class') == 'StaffLines' and el.get('points'):
            pts = [tuple(float(v) for v in p.split(',')) for p in el.get('points').split()]
            lines.append((pts[0][1], min(pts[0][0], pts[-1][0]), max(pts[0][0], pts[-1][0])))
    staves, group = [], []
    for ln in lines:  # MuseScore writes a staff's five lines one after another
        if group and (abs(ln[1] - group[0][1]) > 2 or abs(ln[2] - group[0][2]) > 2 or len(group) == 5):
            group = []
        group.append(ln)
        if len(group) == 5:
            staves.append({'lines': sorted(g[0] for g in group), 'x0': group[0][1], 'x1': group[0][2]})
    staves.sort(key=lambda s: s['lines'][0])
    width = float(root.get('viewBox').split()[2])
    space = float(np.median([(s['lines'][4] - s['lines'][0]) / 4 for s in staves])) if staves else 0.0
    return staves, width, space


def text_boxes(svg: pathlib.Path, classes=('Lyrics', 'InstrumentName')) -> list[dict]:
    """The box of each text MuseScore drew as an outline of a given class,
    in SVG units: a syllable of lyrics, an instrument's name."""
    import re
    out = []
    for el in ET.parse(svg).getroot().iter():
        cls = el.get('class')
        if cls in classes and el.get('d'):
            nums = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', el.get('d'))]
            xs, ys = nums[0::2], nums[1::2]
            if xs and ys:
                out.append({'class': cls, 'box': [min(xs), min(ys), max(xs), max(ys)]})
    return out


def verse_text(part: ET.Element, staff: int, first: int, last: int, verse: str) -> list[str]:
    """The syllables of one verse sung on a staff across measures [first,
    last], in order: '-' after one the next continues, '_' after one held
    over the notes that follow."""
    out = []
    for m in part.findall('measure')[first:last + 1]:
        for n in m.findall('note'):
            if int(n.findtext('staff', '1')) != staff or n.find('chord') is not None:
                continue
            for ly in n.findall('lyric'):
                text = (ly.findtext('text') or '').strip()
                if ly.get('number', '1') == verse and text:
                    # A word carried over from the system before opens with its hyphen.
                    lead = '-' if not out and ly.findtext('syllabic') in ('middle', 'end') else ''
                    out.append(lead + text + ('-' if ly.findtext('syllabic') in ('begin', 'middle') else '')
                               + ('_' if ly.find('extend') is not None else ''))
    return out


def _verses(part: ET.Element, staff: int, first: int, last: int) -> list[str]:
    numbers = []
    for m in part.findall('measure')[first:last + 1]:
        for n in m.findall('note'):
            if int(n.findtext('staff', '1')) == staff:
                for ly in n.findall('lyric'):
                    if (ly.findtext('text') or '').strip() and ly.get('number', '1') not in numbers:
                        numbers.append(ly.get('number', '1'))
    return sorted(numbers, key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else 0, v))


def _systems(root: ET.Element) -> list[list[int]]:
    """For each page, the first measure of each of its systems, as
    MuseScore's own MusicXML marks its layout."""
    pages: list[list[int]] = [[0]]
    for i, m in enumerate(root.find('part').findall('measure')):
        if i == 0:
            continue
        for p in m.findall('print'):
            if p.get('new-page') == 'yes':
                pages.append([i])
            elif p.get('new-system') == 'yes':
                pages[-1].append(i)
    return pages


def _rest_share(part: ET.Element) -> float:
    """The share of a part's measures that hold no note: how much a
    player's part of it would fold into multi-bar rests."""
    measures = part.findall('measure')
    empty = sum(1 for m in measures if not any(n.find('rest') is None for n in m.findall('note')))
    return empty / max(1, len(measures))


def parts_alone(xml: str, rng: random.Random, most: int = 2) -> list[str]:
    """Up to [most] of a score's one-staff parts, each as a score of its
    own, as a player's part is printed: those with many empty measures,
    which fold into multi-bar rests."""
    root = ET.fromstring(xml)
    plist = root.find('part-list')
    chosen = [p for p in root.findall('part') if _rest_share(p) >= 0.25
              and not any(a.findtext('staves') not in (None, '1') for a in p.iter('attributes'))]
    out = []
    for part in rng.sample(chosen, min(most, len(chosen))):
        one = ET.Element('score-partwise', root.attrib)
        for child in root:
            if child.tag not in ('part-list', 'part'):
                one.append(child)
        pl = ET.SubElement(one, 'part-list')
        pl.extend([sp for sp in plist.iter('score-part') if sp.get('id') == part.get('id')])
        one.append(part)
        out.append(ET.tostring(one, encoding='unicode'))
    return out


def cut(name: str, svgs: list[pathlib.Path], exported: pathlib.Path, font: str, rng: random.Random,
        out: pathlib.Path) -> list[dict]:
    """Cut every system's staves of each part out of a score
    MuseScore engraved, with the tokens of what it prints. The tokens come
    from MuseScore's own export of the score, so they say what it drew."""
    root = _root(exported.read_bytes())
    parts = list(_parts(root))
    count = len(parts[0][0].findall('measure'))
    per_system = sum(c for _, c in parts)
    first_staff = np.cumsum([0] + [c for _, c in parts])
    usable = {}  # (part, staff) -> its bars and their states
    for k, (part, c) in enumerate(parts):
        for st in range(1, c + 1):
            # Every staff is drawn, so the first part's first staff tops each system.
            bars, states = tokens.from_part(part, st, top=(k, st) == (0, 1))
            if len(bars) == count:
                usable[(k, st)] = (bars, states)
    names = {sp.get('id'): ((sp.findtext('part-name') or '').strip(), (sp.findtext('part-abbreviation') or '').strip())
             for sp in root.iter('score-part')}
    # The score as MuseScore exported it, kept so later marks can be labeled without engraving again.
    import gzip
    kept = out / 'xml' / name[:2] / f'{name}.musicxml.gz'
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(gzip.compress(exported.read_bytes()))
    lines = []  # text lines: verses of lyrics and instrument names, for the text reader
    pages = _systems(root)
    if len(pages) != len(svgs):
        return []
    samples = []
    for page, (starts, svg) in enumerate(zip(pages, svgs), start=1):
        staves, width, space = staves_of(svg)
        if len(staves) != len(starts) * per_system or not space:
            return samples  # the page is drawn otherwise than its export says
        zoom = rng.uniform(14, 24) / space
        png = svg.with_suffix('.png')
        subprocess.run(['rsvg-convert', '-w', str(int(round(width * zoom))), '-b', 'white', str(svg), '-o', str(png)],
                       check=True)
        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        boxes = text_boxes(svg)
        ends = starts[1:] + [pages[page][0] if page < len(pages) else count]
        for s_i, (first, end) in enumerate(zip(starts, ends)):
            last = end - 1
            lines += _text_lines(name, page, s_i, first, last, parts, names, staves, boxes, s_i * per_system,
                                 first_staff, space, zoom, gray, out, page == 1 and s_i == 0)
            for (k, st), (bars, states) in usable.items():
                if not all(x['readable'] for x in states[first:last + 1]):
                    continue
                staff = staves[s_i * per_system + first_staff[k] + st - 1]
                image = staffcrop.crop(gray, [y * zoom for y in staff['lines']], staff['x0'] * zoom, staff['x1'] * zoom)
                file = out / name[:2] / f'{name}-{page}-{s_i}-{k}-{st}.png'
                file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(file), image)
                samples.append({'image': str(file.relative_to(out)), 'part': k, 'staff': st, 'first': first,
                                'last': last, 'font': font, 'tokens': tokens.stretch(bars, states, first, last).tokens,
                                'xml': str(kept.relative_to(out)), 'top': (k, st) == (0, 1)})
    if lines:
        with open(out / 'text' / 'lines.jsonl', 'a') as f:
            for line in lines:
                f.write(json.dumps({**line, 'font': font}) + '\n')
    return samples


def _text_lines(name, page, s_i, first, last, parts, names, staves, boxes, base, first_staff, space, zoom, gray, out,
                opening) -> list[dict]:
    """Cut a system's text lines for the text reader: each verse sung
    under a staff, labeled with its syllables, kept only where the drawn
    syllables are as many as the score's; and each instrument's name at
    the system's left, labeled with the name the score gives it there."""
    out_lines = []
    system = staves[base:base + int(first_staff[-1])]
    if not system:
        return out_lines
    top, bottom = system[0]['lines'][0], system[-1]['lines'][4]
    lyrics = [b['box'] for b in boxes if b['class'] == 'Lyrics' and top <= (b['box'][1] + b['box'][3]) / 2]
    (out / 'text' / name[:2]).mkdir(parents=True, exist_ok=True)

    def save(img, text, kind, tag):
        if img.size == 0 or not text:
            return
        file = out / 'text' / name[:2] / f'{name}-{page}-{s_i}-{tag}.png'
        cv2.imwrite(str(file), img)
        out_lines.append({'image': str(file.relative_to(out)), 'text': text, 'kind': kind})

    for k, (part, c) in enumerate(parts):
        for st in range(1, c + 1):
            n = int(first_staff[k]) + st - 1
            staff = system[n]
            below_end = system[n + 1]['lines'][0] if n + 1 < len(system) else bottom + 12 * space
            mine = [b for b in lyrics if staff['lines'][4] < (b[1] + b[3]) / 2 < below_end]
            rows: list[list] = []
            for b in sorted(mine, key=lambda b: (b[1] + b[3]) / 2):
                if rows and (b[1] + b[3]) / 2 - (rows[-1][-1][1] + rows[-1][-1][3]) / 2 < 0.8 * space:
                    rows[-1].append(b)
                else:
                    rows.append([b])
            verses = _verses(part, st, first, last)
            if len(rows) != len(verses):
                continue
            for verse, row in zip(verses, rows):
                syllables = verse_text(part, st, first, last, verse)
                if len(syllables) != len(row):
                    continue
                y0, y1 = min(b[1] for b in row) - 0.4 * space, max(b[3] for b in row) + 0.4 * space
                x0, x1 = staff['x0'], staff['x1']
                img = gray[int(y0 * zoom):int(y1 * zoom) + 1, int(x0 * zoom):int(x1 * zoom) + 1]
                save(img, ' '.join(syllables), 'lyrics', f'{k}-{st}-v{verse}')
    for b in (b['box'] for b in boxes if b['class'] == 'InstrumentName'):
        cy = (b[1] + b[3]) / 2
        if not (top - 2 * space <= cy <= bottom + 2 * space):
            continue
        for k, (part, c) in enumerate(parts):
            span = system[int(first_staff[k])]['lines'][0], system[int(first_staff[k]) + c - 1]['lines'][4]
            if span[0] - space <= cy <= span[1] + space:
                full, short = names.get(part.get('id'), ('', ''))
                text = full if opening else short
                img = gray[int((b[1] - 0.3 * space) * zoom):int((b[3] + 0.3 * space) * zoom) + 1,
                           int((b[0] - 0.3 * space) * zoom):int((b[2] + 0.3 * space) * zoom) + 1]
                save(img, text, 'name', f'name{k}-{int(cy)}')
                break
    return out_lines


def render(scores: list[dict], seed: int, out: pathlib.Path, tenor: bool = False,
           quiet_tuplets: bool = False, vocal: bool = False, names: bool = False,
           player_parts: bool = False) -> list[dict]:
    """Engrave a batch of chosen scores in one MuseScore style and cut out
    their staves. MuseScore starts once for the batch; a batch it cannot
    finish is engraved again score by score. With [tenor], parts are
    written in tenor clef where they suit it, and only their staves kept."""
    rng = random.Random(seed)
    font, mss = style(rng, names, multirest=player_parts)
    if quiet_tuplets:  # and for tuplets the file leaves to the style
        mss = mss.replace('  </Style>', '    <tupletNumberType>2</tupletNumberType>\n'
                                         '    <tupletBracketType>2</tupletBracketType>\n  </Style>')
    samples = []
    with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / 'style.mss').write_text(mss)
        jobs = []
        for score in scores:
            name = hashlib.sha1(score['mxl'].encode()).hexdigest()[:12]
            try:
                raw = canonical.load(PDMX / score['mxl'])
                if quiet_tuplets and b'<time-modification' not in raw:
                    continue  # only scores with tuplets
                xml = _clean(raw, tenor, quiet_tuplets, vocal, rng if names else None)
                if tenor and ' tenor="yes"' not in xml or vocal and ' vocal="yes"' not in xml:
                    continue
                if names and int(re.search(r' renamed="(\d+)"', xml).group(1)) < 3:
                    continue  # an ensemble score with at least three named parts
                # A player's part on its own, its empty bars folded into multi-bar rests.
                pieces = [(f'{name}p{k}', x) for k, x in enumerate(parts_alone(xml, rng))] if player_parts \
                    else [(name, xml)]
                for piece, x in pieces:
                    (tmp / f'{piece}.musicxml').write_text(x)
            except Exception:
                continue
            for piece, _ in pieces:
                jobs.append({'in': str(tmp / f'{piece}.musicxml'),
                             'out': [str(tmp / f'{piece}-out.svg'), str(tmp / f'{piece}-out.musicxml')]})
        env = render_mod.mscore_env()

        def run(batch):
            (tmp / 'job.json').write_text(json.dumps(batch))
            subprocess.run([render_mod.MSCORE, '-S', str(tmp / 'style.mss'), '-j', str(tmp / 'job.json')],
                           env=env, capture_output=True, timeout=120 + 30 * len(batch))

        try:
            run(jobs)
        except subprocess.TimeoutExpired:
            pass
        for job in jobs:
            exported = pathlib.Path(job['out'][1])
            if not exported.exists():
                try:
                    run([job])
                except subprocess.TimeoutExpired:
                    continue
            name = pathlib.Path(job['in']).stem
            svgs = sorted(tmp.glob(f'{name}-out-*.svg'), key=lambda p: int(p.stem.rsplit('-', 1)[1]))
            if not exported.exists() or not svgs:
                continue
            try:
                samples += [{**s, 'mxl': next(sc['mxl'] for sc in scores
                                               if hashlib.sha1(sc["mxl"].encode()).hexdigest()[:12] == name[:12])}
                            for s in cut(name, svgs, exported, font, rng, out)
                            if (not tenor or s['tokens'][:1] == ['clef:C4'])
                            and (not vocal or s['tokens'][:1] in (['clef:C1'], ['clef:C3'], ['clef:C4']))]
            except Exception:
                continue
    return samples


def _render_one(args):
    scores, seed, out, tenor, quiet, vocal, names, player_parts = args
    try:
        return render(scores, seed, out, tenor, quiet, vocal, names, player_parts)
    except Exception as e:  # one bad batch never stops the run
        return [{'error': repr(e)[:300]}]


def render_all(name: str, limit: int | None = None, workers: int = 12, seed: int = 0,
               batch: int = 8, chosen: str = 'scores', grand_staff_only: bool = False,
               tenor: bool = False, quiet_tuplets: bool = False, vocal: bool = False,
               names: bool = False, player_parts: bool = False) -> pathlib.Path:
    """Render the chosen scores into DATA/<name>: the staff pictures and
    an index of what each prints, index.jsonl."""
    scores = [json.loads(line) for line in (DATA / f'{chosen}.jsonl').read_text().splitlines()]
    if grand_staff_only:  # scores with a kept part on more than one staff
        scores = [s for s in scores if any(s['staves'][k] > 1 for k in s['parts'])]
    random.Random(seed).shuffle(scores)
    if limit:
        scores = scores[:limit]
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    errors = 0
    batches = [scores[i:i + batch] for i in range(0, len(scores), batch)]
    from .progress import Progress
    bar = Progress(f'render-{name}', len(batches), 'batches', f'engraving {len(scores)} PDMX scores with MuseScore')
    staves = 0
    with open(out / 'index.jsonl', 'w') as index, ProcessPoolExecutor(workers) as pool:
        jobs = [(b, seed * 1_000_003 + i, out, tenor, quiet_tuplets, vocal, names, player_parts)
                for i, b in enumerate(batches)]
        for k, samples in enumerate(pool.map(_render_one, jobs), start=1):
            staves += sum('error' not in s for s in samples)
            bar.update(k, f'{staves} staves, {errors} failed batches')
            for s in samples:
                if 'error' in s:
                    errors += 1
                else:
                    index.write(json.dumps(s) + '\n')
            index.flush()
    (out / 'errors.txt').write_text(f'{errors} batches failed\n')
    bar.finish(f'{staves} staves, {errors} failed batches')
    return out
