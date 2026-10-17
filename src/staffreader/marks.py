"""The marks a staff prints besides its notes, as the reader writes them:
the tokens of the mark inventory chosen on 2026-10-14, and their way to and
from MusicXML.

A note's own marks follow its tokens: its fingering, a notehead of
another shape, a parenthesized accidental. Marks drawn once for a stem
(articulations, ornaments, tremolos, a fermata, an arpeggio, bowings,
slur ends, a glissando, the flag that the note takes a syllable) follow
the last note of its chord. Marks printed above or below the staff stand
before the note where they fall: dynamics, hairpins, octave lines,
pedal marks, segno and coda, and words, spelled a character at a time
between an opening token and ')'. Marks that belong to the whole system
(tempo words and metronome marks, rehearsal marks, first and second
endings) are written on its top staff only.

New marks are only ever new tokens, added at the end of the vocabulary,
so a model trained on these keeps its meaning when more come.

Version 3 (2026-10-15) adds the third group of the inventory: chord
symbols and figured bass, written like words before the note where they
fall, between 'ch(' or 'fb(' and ')'; a tablature note's string and fret
and a shape note's head, after the note; and the Schleifer, on the stem.
Its tokens follow every older one (vocabulary_v3)."""

from __future__ import annotations

import string
import unicodedata
import xml.etree.ElementTree as ET

ARTICULATIONS = ['staccato', 'staccatissimo', 'spiccato', 'accent', 'strong-accent', 'tenuto', 'detached-legato',
                 'soft-accent', 'stress', 'unstress', 'breath-mark', 'caesura', 'scoop', 'plop', 'doit', 'falloff']
ORNAMENTS = ['trill-mark', 'mordent', 'inverted-mordent', 'turn', 'inverted-turn', 'delayed-turn',
             'delayed-inverted-turn', 'vertical-turn', 'shake', 'wavy-line', 'haydn']
TECHNICAL = ['up-bow', 'down-bow', 'harmonic', 'open-string', 'stopped', 'snap-pizzicato', 'thumb-position',
             'double-tongue', 'triple-tongue', 'heel', 'toe']
ACCIDENTAL_MARKS = ['sharp', 'flat', 'natural', 'double-sharp', 'flat-flat']
NOTEHEADS = ['x', 'diamond', 'slash', 'triangle', 'square', 'cross', 'circle-x', 'other']
DYNAMICS = ['pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff', 'sf', 'sfz', 'sffz', 'sfp', 'sfpp',
            'fp', 'fz', 'rf', 'rfz', 'pf', 'n']
OCTAVES = {'8va': ('down', '8'), '8vb': ('up', '8'), '15ma': ('down', '15'), '15mb': ('up', '15')}
BEAT_UNITS = ['whole', 'half', 'quarter', 'eighth', '16th']
# Characters of words, rehearsal marks and endings. Any other is '�'.
CHARS = (' ' + string.ascii_letters + string.digits + '.,;:!?\'"()[]-–—/&+=*#%°'
         + 'àáâãäåæçèéêëìíîïñòóôõöøùúûüýÿßœ' + 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝŒ' + '♭♯♮')
SPACE = '␣'

# Tokens that open text, each closed by ')'.
OPENERS = {'w(': 'words', 'dyn(': 'other dynamics', 'rh(': 'rehearsal', 'mm(': 'metronome', 'end(': 'ending'}

STEM = ([f'm:{a}' for a in ARTICULATIONS] + ['m:art-other'] + [f'm:{o}' for o in ORNAMENTS] + ['m:orn-other']
        + [f'm:accmark-{a}' for a in ACCIDENTAL_MARKS] + [f'm:trem-{n}' for n in range(1, 5)]
        + [f'm:trem2-{n}' for n in range(1, 5)] + ['m:fermata', 'm:arpeggiate', 'm:non-arpeggiate', 'm:gliss']
        + [f'm:{t}' for t in TECHNICAL] + ['sl(', 'sl)', 'ly'])
NOTE = [f'fing:{n}' for n in range(6)] + ['fing:?'] + [f'head:{h}' for h in NOTEHEADS] + ['paren', 'cue']
DIRECTIONS = ([f'dyn:{d}' for d in DYNAMICS] + ['hp<', 'hp>', 'hp.', 'ds(', 'ds.'] + [f'oct:{o}' for o in OCTAVES]
              + ['oct.', 'ped', 'ped.', 'ped^', 'segno', 'coda'] + list(OPENERS) + [f'mm:{u}' for u in BEAT_UNITS]
              + ['end|', 'end~', 'fer|'])
TEXT = [f'c:{SPACE if ch == " " else ch}' for ch in CHARS] + ['c:�', ')']


# Version 3.
SHAPES = ['do', 're', 'mi', 'fa', 'fa up', 'so', 'la', 'ti']  # shape-note heads
KINDS = ['major', 'minor', 'augmented', 'diminished', 'dominant', 'major-seventh', 'minor-seventh',
         'diminished-seventh', 'augmented-seventh', 'half-diminished', 'major-minor', 'major-sixth', 'minor-sixth',
         'dominant-ninth', 'major-ninth', 'minor-ninth', 'dominant-11th', 'major-11th', 'minor-11th',
         'dominant-13th', 'major-13th', 'minor-13th', 'suspended-second', 'suspended-fourth', 'Neapolitan',
         'Italian', 'French', 'German', 'pedal', 'power', 'Tristan', 'other', 'none']
FIGURE_SIGNS = ['sharp', 'flat', 'natural', 'double-sharp', 'flat-flat', 'sharp-sharp', 'slash', 'back-slash', 'plus']
STEM_V3 = ['m:schleifer']
NOTE_V3 = ([f"head:{h.replace(' ', '-')}" for h in SHAPES] + [f'str:{n}' for n in range(1, 10)]
           + [f'fret:{n}' for n in range(0, 31)] + ['fret:?'])
# A chord symbol: its root (cr: and ca:), its kind (ck:), each degree added,
# altered or taken away (cd: with cv: and ca:), and a bass after 'c/'.
CHORD = (['ch('] + [f'cr:{s}' for s in 'ABCDEFG'] + [f'ca:{a}' for a in ('sharp', 'flat', 'double-sharp', 'flat-flat')]
         + [f'ck:{k}' for k in KINDS] + ['cd:add', 'cd:alter', 'cd:subtract'] + [f'cv:{n}' for n in range(1, 14)]
         + ['c/'])
# Figured bass: each figure top to bottom, its prefix, number and suffix, an
# extender line; figures apart by 'f,'.
FIGURES = (['fb('] + [f'fp:{a}' for a in FIGURE_SIGNS] + [f'fn:{n}' for n in range(1, 14)]
           + [f'fs:{a}' for a in FIGURE_SIGNS] + ['fx', 'f,'])
OPENERS_V3 = {'ch(': 'chord symbols', 'fb(': 'figured bass'}
ALL_OPENERS = {**OPENERS, **OPENERS_V3}


def vocabulary() -> list[str]:
    """The mark tokens, in a fixed order, to follow the note tokens."""
    return STEM + NOTE + DIRECTIONS + TEXT


def vocabulary_v3() -> list[str]:
    """The marks version 3 adds, after every older token."""
    return STEM_V3 + NOTE_V3 + CHORD + FIGURES


def text_tokens(opener: str, text: str) -> list[str]:
    text = ' '.join(unicodedata.normalize('NFC', text).split())
    return [opener] + [f'c:{SPACE if ch == " " else (ch if ch in CHARS else "�")}' for ch in text] + [')']


def readable(text: str | None) -> bool:
    """Whether a text is drawn in letters a reader can spell, not only in
    a music font's symbols."""
    text = unicodedata.normalize('NFC', text or '').strip()
    return bool(text) and sum(ch in CHARS for ch in text) >= 0.5 * len(text)


def read_text(toks: list[str]) -> str:
    return ''.join(' ' if t == f'c:{SPACE}' else t[2:] for t in toks if t.startswith('c:'))


# ------------------------------------------------------------ from MusicXML

def note_marks(note: ET.Element) -> tuple[list[str], list[str]]:
    """A MusicXML note's mark tokens: those of the note itself, and those
    drawn once for its stem."""
    own, stem = [], []
    if note.find('cue') is not None:
        own.append('cue')
    acc = note.find('accidental')
    if acc is not None and 'yes' in (acc.get('parentheses'), acc.get('bracket')):
        own.append('paren')
    head = note.findtext('notehead')
    if head and head not in ('normal', 'none'):
        own.append(f"head:{head if head in NOTEHEADS else head.replace(' ', '-') if head in SHAPES else 'other'}")
    if note.find('lyric') is not None and any((ly.findtext('text') or '').strip() for ly in note.findall('lyric')):
        stem.append('ly')
    for notations in note.findall('notations'):
        for child in notations:
            if child.tag == 'articulations':
                stem += [f'm:{g.tag}' if g.tag in ARTICULATIONS else 'm:art-other' for g in child]
            elif child.tag == 'ornaments':
                for g in child:
                    if g.tag == 'tremolo':
                        n = (g.text or '3').strip()
                        n = n if n in '1234' and n else '3'
                        if g.get('type', 'single') == 'single':
                            stem.append(f'm:trem-{n}')
                        elif g.get('type') == 'start':
                            stem.append(f'm:trem2-{n}')
                    elif g.tag == 'wavy-line':
                        if g.get('type') == 'start':
                            stem.append('m:wavy-line')
                    elif g.tag == 'accidental-mark':
                        a = (g.text or '').strip()
                        if a in ACCIDENTAL_MARKS:
                            stem.append(f'm:accmark-{a}')
                    elif g.tag in ORNAMENTS:
                        stem.append(f'm:{g.tag}')
                    elif g.tag == 'schleifer':
                        stem.append('m:schleifer')
                    else:
                        stem.append('m:orn-other')
            elif child.tag == 'technical':
                for g in child:
                    if g.tag == 'fingering':
                        f = (g.text or '').strip()
                        own.append(f'fing:{f}' if f in ('0', '1', '2', '3', '4', '5') else 'fing:?')
                    elif g.tag == 'string' and (g.text or '').strip() in [str(n) for n in range(1, 10)]:
                        own.append(f'str:{g.text.strip()}')
                    elif g.tag == 'fret':
                        f = (g.text or '').strip()
                        own.append(f'fret:{f}' if f.isdigit() and int(f) <= 30 else 'fret:?')
                    elif g.tag in TECHNICAL:
                        stem.append(f'm:{g.tag}')
            elif child.tag == 'slur' and child.get('type') in ('start', 'stop'):
                stem.append('sl(' if child.get('type') == 'start' else 'sl)')
            elif child.tag in ('glissando', 'slide') and child.get('type') == 'start':
                stem.append('m:gliss')
            elif child.tag in ('fermata', 'arpeggiate', 'non-arpeggiate'):
                stem.append(f'm:{child.tag}')
    return own, stem


def direction_tokens(direction: ET.Element, top: bool) -> list[str]:
    """A direction's tokens. Tempo words and metronome marks (a direction
    whose sound sets a tempo) and rehearsal marks belong to the system,
    and are written for its top staff only."""
    if direction.get('print-object') == 'no':
        return []
    sound = direction.find('sound')
    tempo = sound is not None and sound.get('tempo') is not None
    out = []
    for dt in direction.findall('direction-type'):
        for g in dt:
            if g.get('print-object') == 'no':
                continue
            if g.tag == 'dynamics':
                for d in g:
                    if d.tag in DYNAMICS:
                        out.append(f'dyn:{d.tag}')
                    elif (d.text or '').strip():
                        out += text_tokens('dyn(', d.text)
            elif g.tag == 'wedge':
                out += {'crescendo': ['hp<'], 'diminuendo': ['hp>'], 'stop': ['hp.']}.get(g.get('type'), [])
            elif g.tag == 'dashes':
                out += {'start': ['ds('], 'stop': ['ds.']}.get(g.get('type'), [])
            elif g.tag == 'octave-shift':
                kind = g.get('type')
                if kind == 'stop':
                    out.append('oct.')
                else:
                    name = next((k for k, v in OCTAVES.items() if v == (kind, g.get('size', '8'))), None)
                    if name:
                        out.append(f'oct:{name}')
            elif g.tag == 'pedal':
                out += {'start': ['ped'], 'stop': ['ped.'], 'change': ['ped^']}.get(g.get('type'), [])
            elif g.tag in ('segno', 'coda'):
                out.append(g.tag)
            elif g.tag == 'words' and readable(g.text):
                if not tempo or top:
                    out += text_tokens('w(', g.text)
            elif g.tag == 'metronome' and top:
                unit = g.findtext('beat-unit')
                if unit in BEAT_UNITS and (g.findtext('per-minute') or '').strip():
                    out += [f'mm:{unit}'] + ['.'] * len(g.findall('beat-unit-dot'))
                    out += text_tokens('mm(', g.findtext('per-minute'))
            elif g.tag == 'rehearsal' and top and (g.text or '').strip():
                out += text_tokens('rh(', g.text)
    return out


_ALTERS = {1: 'sharp', -1: 'flat', 2: 'double-sharp', -2: 'flat-flat'}


def _step_tokens(step: str | None, alter: str | None) -> list[str]:
    out = [f'cr:{step}'] if step in list('ABCDEFG') else []
    try:
        a = int(float(alter or 0))
    except ValueError:
        a = 0
    return out + ([f'ca:{_ALTERS[a]}'] if a in _ALTERS else [])


def harmony_tokens(h: ET.Element) -> list[str]:
    """A chord symbol's tokens: root, kind, degrees and bass. One drawn as
    a fretboard alone, or hidden, has none."""
    if h.get('print-object') == 'no':
        return []
    root = h.find('root')
    if root is None:
        return []
    kind = (h.findtext('kind') or 'major').strip()
    out = ['ch('] + _step_tokens(root.findtext('root-step'), root.findtext('root-alter'))
    out.append(f"ck:{kind if kind in KINDS else 'other'}")
    for d in h.findall('degree'):
        what = d.findtext('degree-type')
        value = (d.findtext('degree-value') or '').strip()
        if what in ('add', 'alter', 'subtract') and value.isdigit() and 1 <= int(value) <= 13:
            out += [f'cd:{what}', f'cv:{value}']
            a = int(float(d.findtext('degree-alter') or 0))
            if a in _ALTERS:
                out.append(f'ca:{_ALTERS[a]}')
    bass = h.find('bass')
    if bass is not None:
        out += ['c/'] + _step_tokens(bass.findtext('bass-step'), bass.findtext('bass-alter'))
    return out + [')']


def figure_tokens(fb: ET.Element) -> list[str]:
    """A figured bass's tokens, its figures top to bottom."""
    if fb.get('print-object') == 'no':
        return []
    out = ['fb(']
    for k, f in enumerate(fb.findall('figure')):
        if k:
            out.append('f,')
        pre, num, suf = (f.findtext(t) for t in ('prefix', 'figure-number', 'suffix'))
        if pre in FIGURE_SIGNS:
            out.append(f'fp:{pre}')
        if (num or '').strip().isdigit() and 1 <= int(num) <= 13:
            out.append(f'fn:{int(num)}')
        if suf in FIGURE_SIGNS:
            out.append(f'fs:{suf}')
        if f.find('extend') is not None and f.find('extend').get('type', 'start') != 'stop':
            out.append('fx')
    return out + [')'] if len(out) > 1 else []


def ending_tokens(barline: ET.Element) -> list[str]:
    """An ending's start (with its number as printed) or its end."""
    out = []
    for e in barline.findall('ending'):
        kind = e.get('type')
        if kind == 'start':
            out += text_tokens('end(', (e.text or '').strip() or f"{e.get('number', '1')}.")
        elif kind == 'stop':
            out.append('end|')
        elif kind == 'discontinue':
            out.append('end~')
    return out


# -------------------------------------------------------------- to MusicXML

JUMPS = [('d.c.', 'dacapo'), ('da capo', 'dacapo'), ('d.s.', 'dalsegno'), ('dal segno', 'dalsegno'),
         ('to coda', 'tocoda'), ('fine', 'fine')]


def write_note_marks(el: ET.Element, marks: list[str]):
    """Put a read note's marks into its MusicXML note."""
    if 'cue' in marks:  # after a grace, before everything else
        el.insert(1 if len(el) and el[0].tag == 'grace' else 0, ET.Element('cue'))
    acc = el.find('accidental')
    if 'paren' in marks and acc is not None:
        acc.set('parentheses', 'yes')
    heads = [m[5:] for m in marks if m.startswith('head:')]
    if heads and heads[0] != 'other':
        h = ET.Element('notehead')
        h.text = heads[0].replace('-', ' ') if heads[0].replace('-', ' ') in SHAPES else heads[0]
        after = ('staff', 'beam', 'notations', 'lyric')
        el.insert(next((k for k, c in enumerate(el) if c.tag in after), len(el)), h)
    groups: dict[str, ET.Element] = {}
    notations = el.find('notations')

    def group(name: str) -> ET.Element:
        nonlocal notations
        if notations is None:
            notations = ET.SubElement(el, 'notations')
        if name not in groups:
            groups[name] = ET.SubElement(notations, name)
        return groups[name]

    for m in marks:
        if m in ('sl(', 'sl)'):
            if notations is None:
                notations = ET.SubElement(el, 'notations')
            ET.SubElement(notations, 'slur', type='start' if m == 'sl(' else 'stop')
        elif m.startswith('fing:'):
            ET.SubElement(group('technical'), 'fingering').text = m[5:] if m != 'fing:?' else ''
        elif m.startswith('str:'):
            ET.SubElement(group('technical'), 'string').text = m[4:]
        elif m.startswith('fret:') and m != 'fret:?':
            ET.SubElement(group('technical'), 'fret').text = m[5:]
        elif not m.startswith('m:'):
            continue
        else:
            name = m[2:]
            if name in ARTICULATIONS:
                ET.SubElement(group('articulations'), name)
            elif name == 'art-other':
                ET.SubElement(group('articulations'), 'other-articulation')
            elif name == 'wavy-line':
                # The token marks where a trill line starts, not where it
                # ends; a line left open loses the score's other trills in
                # professional engraving software, so it ends on its note.
                ET.SubElement(group('ornaments'), name, type='start')
                ET.SubElement(group('ornaments'), name, type='stop')
            elif name in ORNAMENTS:
                ET.SubElement(group('ornaments'), name)
            elif name == 'orn-other':
                ET.SubElement(group('ornaments'), 'other-ornament')
            elif name == 'schleifer':
                ET.SubElement(group('ornaments'), 'schleifer')
            elif name.startswith('accmark-'):
                ET.SubElement(group('ornaments'), 'accidental-mark').text = name[8:]
            elif name.startswith('trem-'):
                ET.SubElement(group('ornaments'), 'tremolo', type='single').text = name[5:]
            elif name.startswith('trem2-'):
                ET.SubElement(group('ornaments'), 'tremolo', type='start').text = name[6:]
            elif name in TECHNICAL:
                ET.SubElement(group('technical'), name)
            elif name in ('fermata', 'arpeggiate', 'non-arpeggiate'):
                if notations is None:
                    notations = ET.SubElement(el, 'notations')
                ET.SubElement(notations, name)
            elif name == 'gliss':
                if notations is None:
                    notations = ET.SubElement(el, 'notations')
                ET.SubElement(notations, 'glissando', type='start')


def write_direction(m: ET.Element, toks: list[str], staff: int | None):
    """MusicXML directions for a run of read direction tokens."""
    i = 0
    while i < len(toks):
        t = toks[i]
        d = ET.Element('direction')
        dt = ET.SubElement(d, 'direction-type')
        jump = None
        if t in OPENERS_V3:
            j = toks.index(')', i) if ')' in toks[i:] else len(toks)
            el = (write_harmony if t == 'ch(' else write_figures)(toks[i + 1:j])
            i = j + 1
            if el is not None:
                if staff is not None and t == 'ch(':
                    ET.SubElement(el, 'staff').text = str(staff)
                m.append(el)
            continue
        if t in OPENERS:
            j = toks.index(')', i) if ')' in toks[i:] else len(toks)
            text = read_text(toks[i + 1:j])
            i = j + 1
            if t == 'w(':
                ET.SubElement(dt, 'words').text = text
                low = text.lower()
                jump = next((v for k, v in JUMPS if k in low), None)
            elif t == 'dyn(':
                ET.SubElement(ET.SubElement(dt, 'dynamics'), 'other-dynamics').text = text
            elif t == 'rh(':
                ET.SubElement(dt, 'rehearsal').text = text
            else:
                continue  # metronome digits come with their unit; endings belong to the barline
        else:
            i += 1
            jump = None
            if t.startswith('dyn:'):
                ET.SubElement(ET.SubElement(dt, 'dynamics'), t[4:])
            elif t in ('hp<', 'hp>', 'hp.'):
                ET.SubElement(dt, 'wedge', type={'hp<': 'crescendo', 'hp>': 'diminuendo', 'hp.': 'stop'}[t])
            elif t in ('ds(', 'ds.'):
                ET.SubElement(dt, 'dashes', type='start' if t == 'ds(' else 'stop')
            elif t.startswith('oct:'):
                kind, size = OCTAVES.get(t[4:], ('down', '8'))
                ET.SubElement(dt, 'octave-shift', type=kind, size=size)
            elif t == 'oct.':
                ET.SubElement(dt, 'octave-shift', type='stop')
            elif t in ('ped', 'ped.', 'ped^'):
                ET.SubElement(dt, 'pedal', type={'ped': 'start', 'ped.': 'stop', 'ped^': 'change'}[t])
            elif t in ('segno', 'coda'):
                ET.SubElement(dt, t)
            elif t.startswith('mm:'):
                dots = 0
                while i < len(toks) and toks[i] == '.':
                    dots += 1
                    i += 1
                per = ''
                if i < len(toks) and toks[i] == 'mm(':
                    j = toks.index(')', i) if ')' in toks[i:] else len(toks)
                    per = read_text(toks[i + 1:j])
                    i = j + 1
                met = ET.SubElement(dt, 'metronome')
                ET.SubElement(met, 'beat-unit').text = t[3:]
                for _ in range(dots):
                    ET.SubElement(met, 'beat-unit-dot')
                ET.SubElement(met, 'per-minute').text = per
            else:
                continue
        if staff is not None:
            ET.SubElement(d, 'staff').text = str(staff)
        if t == 'w(' and jump:  # MusicXML puts the sound after the staff
            ET.SubElement(d, 'sound', **{jump: 'yes' if jump != 'tocoda' else 'coda'})
        m.append(d)


_ALTER_OF = {v: k for k, v in _ALTERS.items()}


def write_harmony(toks: list[str]) -> ET.Element | None:
    """A MusicXML harmony from a chord symbol's tokens."""
    h = ET.Element('harmony')
    part, last = h, None
    root = None
    for t in toks:
        if t.startswith('cr:'):
            if part is h:
                root = ET.SubElement(h, 'root')
                ET.SubElement(root, 'root-step').text = t[3:]
                last = root
            else:
                ET.SubElement(part, 'bass-step').text = t[3:]
                last = part
        elif t.startswith('ca:') and last is not None:
            a = str(_ALTER_OF.get(t[3:], 0))
            tag = {'root': 'root-alter', 'bass': 'bass-alter', 'degree': 'degree-alter'}[last.tag]
            ET.SubElement(last, tag).text = a
        elif t.startswith('ck:'):
            ET.SubElement(h, 'kind').text = t[3:]
        elif t.startswith('cd:'):
            last = ET.SubElement(h, 'degree')
            last.set('_type', t[3:])
        elif t.startswith('cv:') and last is not None and last.tag == 'degree':
            ET.SubElement(last, 'degree-value').text = t[3:]
        elif t == 'c/':
            part = ET.SubElement(h, 'bass')
            last = part
    if root is None:
        return None
    if h.find('kind') is None:
        ET.SubElement(h, 'kind').text = 'major'
    kind = h.find('kind')  # MusicXML orders root, kind, bass, degree
    for el in [kind] + h.findall('bass') + h.findall('degree'):
        h.remove(el)
        h.append(el)
    for d in h.findall('degree'):
        alter = d.find('degree-alter')
        if alter is None:
            alter = ET.SubElement(d, 'degree-alter')
            alter.text = '0'
        else:
            d.remove(alter)
            d.append(alter)
        ET.SubElement(d, 'degree-type').text = d.attrib.pop('_type')
    return h


def write_figures(toks: list[str]) -> ET.Element | None:
    """A MusicXML figured bass from its tokens."""
    fb = ET.Element('figured-bass')
    fig = None
    for t in toks + ['f,']:
        if t == 'f,':
            if fig is not None and len(fig):
                order = {'prefix': 0, 'figure-number': 1, 'suffix': 2, 'extend': 3}
                for el in sorted(list(fig), key=lambda e: order[e.tag]):
                    fig.remove(el)
                    fig.append(el)
                fb.append(fig)
            fig = ET.Element('figure')
            continue
        if fig is None:
            fig = ET.Element('figure')
        if t.startswith('fp:'):
            ET.SubElement(fig, 'prefix').text = t[3:]
        elif t.startswith('fn:'):
            ET.SubElement(fig, 'figure-number').text = t[3:]
        elif t.startswith('fs:'):
            ET.SubElement(fig, 'suffix').text = t[3:]
        elif t == 'fx':
            ET.SubElement(fig, 'extend')
    return fb if len(fb) else None


# The direction tokens and text openers of each family the harness scores
# (metrics.family), for dropping families an answer key does not hold.
FAMILY_TOKENS = {
    'dynamics': ('dyn:', 'dyn('),
    'hairpins': ('hp<', 'hp>', 'hp.'),
    'pedal': ('ped', 'ped.', 'ped^'),
    'tempo marks': ('mm:', 'mm('),
    'words': ('w(', 'ds(', 'ds.'),
    'chord symbols': ('ch(',),
    'figured bass': ('fb(',),
}


def drop(toks: list[str], families) -> list[str]:
    """A staff's tokens without the marks of [families]: their direction
    tokens, and their texts with their characters (a metronome mark with
    its dots)."""
    gone = tuple(t for f in families for t in FAMILY_TOKENS.get(f, ()))
    out, skipping, metronome = [], False, False
    for t in toks:
        if skipping:
            skipping = t != ')'
            continue
        if metronome and t == '.':
            continue
        metronome = False
        if t in ALL_OPENERS and t in gone:
            skipping = True
            continue
        if t in gone or any(g.endswith(':') and t.startswith(g) for g in gone):
            metronome = t.startswith('mm:')
            continue
        out.append(t)
    return out


def strip(toks: list[str]) -> list[str]:
    """A staff's tokens without its marks: the notes alone, as format 1
    wrote them. Words and the like go with their characters, and a
    metronome mark with its dots."""
    marks = set(vocabulary()) | set(vocabulary_v3())
    out, skipping, metronome = [], False, False
    for t in toks:
        if skipping:
            skipping = t != ')'
            continue
        if t in ALL_OPENERS:
            skipping = True
            continue
        if metronome and t == '.':
            continue
        metronome = t.startswith('mm:')
        if t not in marks:
            out.append(t)
    return out
