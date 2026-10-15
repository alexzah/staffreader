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
so a model trained on these keeps its meaning when more come."""

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


def vocabulary() -> list[str]:
    """The mark tokens, in a fixed order, to follow the note tokens."""
    return STEM + NOTE + DIRECTIONS + TEXT


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
        own.append(f"head:{head if head in NOTEHEADS else 'other'}")
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
                    else:
                        stem.append('m:orn-other')
            elif child.tag == 'technical':
                for g in child:
                    if g.tag == 'fingering':
                        f = (g.text or '').strip()
                        own.append(f'fing:{f}' if f in ('0', '1', '2', '3', '4', '5') else 'fing:?')
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
        h.text = heads[0]
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
        elif not m.startswith('m:'):
            continue
        else:
            name = m[2:]
            if name in ARTICULATIONS:
                ET.SubElement(group('articulations'), name)
            elif name == 'art-other':
                ET.SubElement(group('articulations'), 'other-articulation')
            elif name in ORNAMENTS:
                ET.SubElement(group('ornaments'), name, **({'type': 'start'} if name == 'wavy-line' else {}))
            elif name == 'orn-other':
                ET.SubElement(group('ornaments'), 'other-ornament')
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


# The direction tokens and text openers of each family the harness scores
# (metrics.family), for dropping families an answer key does not hold.
FAMILY_TOKENS = {
    'dynamics': ('dyn:', 'dyn('),
    'hairpins': ('hp<', 'hp>', 'hp.'),
    'pedal': ('ped', 'ped.', 'ped^'),
    'tempo marks': ('mm:', 'mm('),
    'words': ('w(', 'ds(', 'ds.'),
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
        if t in OPENERS and t in gone:
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
    marks = set(vocabulary())
    out, skipping, metronome = [], False, False
    for t in toks:
        if skipping:
            skipping = t != ')'
            continue
        if t in OPENERS:
            skipping = True
            continue
        if metronome and t == '.':
            continue
        metronome = t.startswith('mm:')
        if t not in marks:
            out.append(t)
    return out
