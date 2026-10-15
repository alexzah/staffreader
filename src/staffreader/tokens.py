"""A staff's music as the reader writes it: a sequence of tokens, one
system's stretch of one staff at a time, the way the page prints it.

A stretch begins with what every system reprints, the clef and key, and
the time where the page prints one. Then each bar's notes and rests in
order, and the barline that ends the bar. A note is its printed
accidental, if any, then its written pitch, then its look: its type,
dots, tuplet, tie, grace. A note struck with the one before it is
marked as a chord. Pitches are written as printed, a step and an octave,
so the alteration comes from the key and the accidentals, as a reader of
the page works it out.

    clef:G2 key:-1 time:3/4
    N:D5 d:quarter  acc:sharp N:C5 d:eighth  N:B4 d:eighth  ... |
    R d:half  N:A4 d:quarter :|

`from_musicxml` turns a score we hold into stretches of tokens, and
`to_musicxml` turns read tokens back into MusicXML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction

TYPES = ['breve', 'whole', 'half', 'quarter', 'eighth', '16th', '32nd', '64th']
TYPE_LENGTH = {'breve': Fraction(8), 'whole': Fraction(4), 'half': Fraction(2), 'quarter': Fraction(1),
               'eighth': Fraction(1, 2), '16th': Fraction(1, 4), '32nd': Fraction(1, 8), '64th': Fraction(1, 16)}
ACCIDENTALS = {'sharp': 1, 'flat': -1, 'natural': 0, 'double-sharp': 2, 'sharp-sharp': 2, 'flat-flat': -2}
BARLINES = ['|', '||', '|]', ':|', '|:', ':|:']
# Tuplets as actual:normal notes, the ones common enough to learn.
TUPLETS = ['3:2', '5:4', '6:4', '7:4', '7:8', '9:8', '2:3', '4:3', '4:6', '6:8', '5:2', '5:6']
STEPS = 'CDEFGAB'
SHARPS = 'FCGDAEB'


def vocabulary() -> list[str]:
    """Every token the reader may write, in a fixed order."""
    words = ['<pad>', '<s>', '</s>', 'R', 'R:measure', '.', '~', 'gr', '+']
    words += [f'tup:{t}' for t in TUPLETS]
    words += [f'clef:{c}' for c in ('G2', 'F4', 'C3', 'C4', 'C1', 'G2-8', 'G2+8', 'F3', 'C2', 'F4-8', 'percussion',
                                     'F4+8', 'G1', 'F5', 'C5')]
    words += [f'key:{k}' for k in range(-7, 8)]
    words += ['time:C', 'time:C|'] + [f'time:{b}/{t}' for t in (1, 2, 4, 8, 16) for b in range(1, 19)]
    words += [f'acc:{a}' for a in ('sharp', 'flat', 'natural', 'double-sharp', 'flat-flat')]
    words += [f'N:{s}{o}' for o in range(0, 9) for s in STEPS]
    words += [f'd:{t}' for t in TYPES]
    words += BARLINES
    words += ['&', 'S']  # the next voice of the bar; space a voice leaves blank
    from . import marks
    words += marks.vocabulary()  # version 2: the marks of the inventory (marks.py)
    words += ['task:notes', 'task:marks']  # what a reading is to hold: the notes alone, or the marks too
    return words


@dataclass
class Stretch:
    """One staff's music across one system's bars."""
    tokens: list[str]
    first: int  # the first measure's index in the part, from 0
    last: int


# ------------------------------------------------------------ from MusicXML

def _clef(c: ET.Element) -> str:
    sign = c.findtext('sign', 'G')
    if sign == 'percussion':
        return 'percussion'
    clef = f"{sign}{c.findtext('line', '')}"
    octave = c.findtext('clef-octave-change')
    if octave and octave != '0':
        clef += '+8' if int(octave) > 0 else '-8'
    return clef


def _time(t: ET.Element) -> str:
    symbol = t.get('symbol')
    beats, beat_type = t.findtext('beats'), t.findtext('beat-type')
    if symbol == 'common' and (beats, beat_type) == ('4', '4'):
        return 'C'
    if symbol == 'cut' and (beats, beat_type) == ('2', '2'):
        return 'C|'
    return f'{beats}/{beat_type}'


def _barline(measure: ET.Element, nxt: ET.Element | None) -> str:
    """The barline printed at the end of [measure]: its own right barline,
    with a repeat that starts the next bar drawn on the same line."""
    right = next((b for b in measure.findall('barline') if b.get('location', 'right') == 'right'), None)
    left_next = next((b for b in nxt.findall('barline') if b.get('location') == 'left'), None) if nxt is not None else None
    end = right is not None and right.find('repeat') is not None and right.find('repeat').get('direction') == 'backward'
    start = left_next is not None and left_next.find('repeat') is not None and \
        left_next.find('repeat').get('direction') == 'forward'
    if end and start:
        return ':|:'
    if end:
        return ':|'
    if start:
        return '|:'
    style = right.findtext('bar-style') if right is not None else None
    if style == 'light-heavy':
        return '|]'
    if style in ('light-light', 'heavy-heavy', 'heavy-light'):
        return '||'
    return '|'


def _note_tokens(n: ET.Element, divisions: int) -> list[str] | None:
    """A note's or rest's tokens, or None for what the page does not show."""
    if n.get('print-object') == 'no':
        return ['?']
    out = []
    if n.find('chord') is not None:
        out.append('+')
    if n.find('grace') is not None:
        out.append('gr')
    rest = n.find('rest')
    typ = n.findtext('type')
    if rest is not None:
        if rest.get('measure') == 'yes' or (typ is None and n.find('grace') is None):
            return ['R:measure']
        out.append('R')
    else:
        acc = n.findtext('accidental')
        if acc in ACCIDENTALS:
            out.append(f"acc:{'double-sharp' if acc == 'sharp-sharp' else acc}")
        p = n.find('pitch')
        if p is None:
            p = n.find('unpitched')
            if p is None:
                return None
            out.append(f"N:{p.findtext('display-step', 'B')}{p.findtext('display-octave', '4')}")
        else:
            out.append(f"N:{p.findtext('step')}{p.findtext('octave')}")
    if typ is None:
        duration = Fraction(int(float(n.findtext('duration', '0'))), divisions)
        typ = next((t for t in TYPES if TYPE_LENGTH[t] == duration), 'quarter')
    if typ not in TYPE_LENGTH:
        typ = {'128th': '64th', 'long': 'breve', 'maxima': 'breve', '256th': '64th'}.get(typ, 'quarter')
    out.append(f'd:{typ}')
    out += ['.'] * len(n.findall('dot'))
    tm = n.find('time-modification')
    if tm is not None:
        ratio = f"{tm.findtext('actual-notes')}:{tm.findtext('normal-notes')}"
        if ratio not in TUPLETS:
            return ['?']
        out.append(f'tup:{ratio}')
    if any(t.get('type') == 'start' for t in n.findall('tie')):
        out.append('~')
    if n.find('grace') is None and n.find('chord') is None:
        shown = {'type': typ, 'dots': out.count('.'), 'tuplet': next((t[4:] for t in out if t.startswith('tup:')), None)}
        if length(shown) != Fraction(int(float(n.findtext('duration', '0'))), divisions):
            return ['?']  # the look does not give the length
    return out


def from_part(part: ET.Element, staff: int = 1, top: bool = True) -> tuple[list[list[str]], list[dict]]:
    """Each measure's tokens for one staff of a part, and for each measure
    what is in force where its music starts, what it prints before its
    first note, and how many voices its staff holds. Marks that belong to
    the whole system (tempo, rehearsal marks, endings) are written only
    where the staff is the system's [top] one."""
    from . import marks as M
    divisions = 1
    clef, key, time = 'G2', 0, None
    measures = part.findall('measure')
    bars, states = [], []
    for i, m in enumerate(measures):
        opening: dict[str, str] = {}
        start = None
        at_start = (clef, key, time)  # in force where the bar begins
        streams: dict[str, list[str]] = {}  # each voice's tokens in this bar
        reached: dict[str, Fraction] = {}  # how far into the bar each voice has got
        inline: list[str] = []  # a change printed mid-bar, waiting for the next note
        starts: dict[str, list[tuple[int, Fraction]]] = {}  # where each voice's items begin, and when
        stem: dict[str, list[str]] = {}  # marks for the stem of each voice's last chord, not yet written
        directions: list[tuple[Fraction, list[str]]] = []  # marks above or below the staff, by onset
        opening_marks: list[str] = []  # an ending that starts with the bar
        closing_marks: list[str] = []  # an ending that stops with it, a fermata on its barline
        pos = last = Fraction(0)
        unreadable = m.find('.//multiple-rest') is not None
        for el in m:
            if el.tag == 'attributes':
                if (d := el.findtext('divisions')) is not None:
                    divisions = int(float(d))
                changed = []
                for c in el.findall('clef'):
                    if int(c.get('number', '1')) == staff:
                        new = _clef(c)
                        if new != clef or i == 0:
                            changed.append(('clef', f'clef:{new}'))
                        clef = new
                k = el.find('key')
                if k is not None and k.findtext('fifths') is not None:
                    new = int(k.findtext('fifths'))
                    if new != key or i == 0:
                        changed.append(('key', f'key:{new}'))
                    key = new
                t = el.find('time')
                if t is not None and t.findtext('beats') and '+' not in t.findtext('beats'):
                    new = _time(t)
                    if new != time or i == 0:
                        changed.append(('time', f'time:{new}'))
                    time = new
                for kind, tok in changed:
                    # Only a change at the bar's very start opens it. In a
                    # grand staff the lower staff's mid-bar clef change comes
                    # after the upper staff's notes, before any of its own.
                    if start is None and pos == 0:
                        opening[kind] = tok
                        at_start = (clef, key, time)
                    else:  # a change in mid-bar is printed where it falls
                        inline.append(tok)
            elif el.tag in ('backup', 'forward'):
                step = Fraction(int(float(el.findtext('duration', '0'))), divisions)
                pos += step if el.tag == 'forward' else -step
            elif el.tag == 'direction':
                if int(el.findtext('staff', '1')) == staff:
                    toks = M.direction_tokens(el, top)
                    if toks:
                        directions.append((pos + Fraction(int(float(el.findtext('offset', '0'))), divisions), toks))
            elif el.tag == 'barline':
                if top:
                    ending = M.ending_tokens(el)
                    if el.get('location') == 'left':
                        opening_marks += [t for t in ending if t not in ('end|', 'end~')]
                    else:
                        closing_marks += [t for t in ending if t in ('end|', 'end~')]
                        opening_marks += [t for t in ending if t not in ('end|', 'end~')]
                if el.find('fermata') is not None and el.get('location', 'right') == 'right':
                    closing_marks.append('fer|')
            elif el.tag == 'note':
                grace = el.find('grace') is not None
                chord = el.find('chord') is not None
                length_ = Fraction(0) if grace else Fraction(int(float(el.findtext('duration', '0'))), divisions)
                onset = last if chord else pos
                if not chord and not grace:
                    pos += length_
                if not chord:
                    last = onset
                if int(el.findtext('staff', '1')) != staff:
                    continue
                if el.get('print-object') == 'no':
                    continue  # a hidden note is space the page leaves blank
                nt = _note_tokens(el, divisions)
                if nt == ['?']:
                    unreadable = True
                    continue
                if not nt:
                    continue
                v = el.findtext('voice', '1')
                stream = streams.setdefault(v, [])
                if start is None:
                    start = at_start
                own, stem_marks = M.note_marks(el)
                if not chord:
                    stream += _ordered(stem.pop(v, []))  # the last chord's stem marks follow it
                    gap = onset - reached.get(v, Fraction(0))
                    if gap < 0:
                        unreadable = True  # a voice running over itself
                    elif gap > 0:
                        space = _space(gap)
                        if space is None:
                            unreadable = True
                        else:
                            stream += space
                    reached[v] = onset + length_
                    starts.setdefault(v, []).append((len(stream), onset))
                stream += inline
                inline = []
                stream += nt + own
                stem.setdefault(v, [])
                stem[v] += [t for t in stem_marks if t not in stem[v]]
        if start is None:
            start = at_start
        order = sorted(streams, key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else 0, v))
        for v in order:
            streams[v] += _ordered(stem.pop(v, []))
        tail = _place(directions, streams, starts, order)
        tokens: list[str] = list(opening_marks)
        for k, v in enumerate(order):
            if k:
                tokens.append('&')
            tokens += streams[v]
            if k == 0:
                tokens += tail
        if not order:
            tokens += tail
        tokens += inline + closing_marks
        tokens.append(_barline(m, measures[i + 1] if i + 1 < len(measures) else None))
        bars.append(tokens)
        states.append({'clef': start[0], 'key': start[1], 'time': start[2], 'voices': len(streams),
                       'opening': [opening[k] for k in ('clef', 'key', 'time') if k in opening],
                       'readable': not unreadable and len(streams) <= 4})
    return bars, states


def _ordered(stem: list[str]) -> list[str]:
    """A stem's marks in the vocabulary's order, so a chord's marks read
    the same whichever of its notes carried them."""
    from . import marks as M
    rank = {t: k for k, t in enumerate(M.STEM)}
    return sorted(stem, key=lambda t: rank.get(t, len(rank)))


def _place(directions: list[tuple[Fraction, list[str]]], streams: dict[str, list[str]],
           starts: dict[str, list[tuple[int, Fraction]]], order: list[str]) -> list[str]:
    """Put each direction before the first note or rest the staff shows
    at or after it, in the first voice that has one there, as the answer
    key places it. Those after every note are returned, to end the bar's
    first voice."""
    tail: list[str] = []
    inserts: dict[str, list[tuple[int, list[str]]]] = {}
    for onset, toks in directions:
        later = [o for v in order for _, o in starts.get(v, []) if o >= onset]
        if not later:
            tail += toks
            continue
        at = min(later)
        v = next(v for v in order if any(o == at for _, o in starts.get(v, [])))
        index = next(k for k, o in starts[v] if o == at)
        inserts.setdefault(v, []).append((index, toks))
    for v, items in inserts.items():
        # From the end, so earlier places keep their index; at one place,
        # in the order they came.
        for _, (index, toks) in sorted(enumerate(items), key=lambda x: (-x[1][0], -x[0])):
            streams[v][index:index] = toks
    return tail


def _space(gap: Fraction) -> list[str] | None:
    """A blank stretch of a voice as space tokens, longest types first, or
    None where no plain types add up to it."""
    out = []
    for t in TYPES:
        while gap >= TYPE_LENGTH[t]:
            out += ['S', f'd:{t}']
            gap -= TYPE_LENGTH[t]
    return out if gap == 0 else None


def stretch(bars: list[list[str]], states: list[dict], first: int, last: int) -> Stretch:
    """The tokens of measures [first, last] as one system prints them: the
    clef and key it opens with, the time where it is printed, then the
    bars. A clef, key or time changed at a bar inside the system is
    printed there."""
    s0 = states[first]
    head = [f"clef:{s0['clef']}", f"key:{s0['key']}"]
    if any(t.startswith('time:') for t in s0['opening']) and s0['time']:
        head.append(f"time:{s0['time']}")
    body = []
    for i in range(first, last + 1):
        if i > first:
            body += states[i]['opening']
        body += bars[i]
    return Stretch(head + body, first, last)


# -------------------------------------------------------------- to MusicXML

DIVISIONS = 5040  # holds every tuplet of every type exactly


@dataclass
class Bar:
    clef: str | None = None
    key: int | None = None
    time: str | None = None
    notes: list[dict] = field(default_factory=list)
    barline: str = '|'
    ending: str | None = None  # an ending that starts with the bar, as printed: '1.'
    ending_end: str | None = None  # 'end|' or 'end~' where one stops with the bar
    fermata: bool = False  # a fermata on the bar's barline
    after: list[str] = field(default_factory=list)  # marks after the first voice's last note


def parse(tokens: list[str]) -> list[Bar]:
    """Read tokens back into bars, each note with its printed look and
    marks, and the marks above or below the staff before the note where
    they fall."""
    from . import marks as M
    stem, own = set(M.STEM), set(M.NOTE)
    bars = [Bar()]
    pending: dict = {}
    voice = 0
    waiting: list[str] = []  # marks above or below the staff, for the next note
    text: list[str] | None = None  # an open run of characters, with its opener
    metronome = False  # a metronome mark's dots come before its number

    def flush_after(bar: Bar):
        nonlocal waiting
        if voice == 0:
            bar.after += waiting
        waiting = []

    for tok in tokens:
        bar = bars[-1]
        if text is not None:
            text.append(tok)
            if tok == ')':
                if text[0] == 'end(':
                    bar.ending = M.read_text(text[1:-1])
                else:
                    waiting += text
                text = None
                metronome = False
            continue
        if tok in M.OPENERS:
            text = [tok]
            continue
        if metronome and tok == '.':
            waiting.append(tok)
            continue
        if tok in BARLINES:
            flush_after(bar)
            bar.barline = tok
            bars.append(Bar())
            voice = 0
            continue
        if tok in ('end|', 'end~'):
            bar.ending_end = tok
            continue
        if tok == 'fer|':
            bar.fermata = True
            continue
        if tok in M.DIRECTIONS:
            waiting.append(tok)
            metronome = tok.startswith('mm:')
            continue
        if tok in stem or tok in own:
            if bar.notes:
                bar.notes[-1].setdefault('marks', []).append(tok)
            continue
        if tok == '&':
            flush_after(bar)
            voice += 1
            pending = {}
            continue
        if tok.startswith('clef:'):
            bar.clef = tok[5:]
        elif tok.startswith('key:'):
            bar.key = int(tok[4:])
        elif tok.startswith('time:'):
            bar.time = tok[5:]
        elif tok == '+':
            pending['chord'] = True
        elif tok == 'gr':
            pending['grace'] = True
        elif tok.startswith('acc:'):
            pending['accidental'] = tok[4:]
        elif tok in ('R', 'R:measure', 'S') or tok.startswith('N:'):
            note = {'rest': tok.startswith('R'), 'measure': tok == 'R:measure', 'space': tok == 'S',
                    'pitch': tok[2:] if tok.startswith('N:') else None, 'type': 'quarter', 'dots': 0,
                    'tuplet': None, 'tie': False, 'voice': voice, **pending}
            if waiting and tok != 'S':
                note['before'] = waiting
                waiting = []
            pending = {}
            bar.notes.append(note)
        elif tok.startswith('d:') and bar.notes:
            bar.notes[-1]['type'] = tok[2:]
        elif tok == '.' and bar.notes:
            bar.notes[-1]['dots'] += 1
        elif tok.startswith('tup:') and bar.notes:
            bar.notes[-1]['tuplet'] = tok[4:]
        elif tok == '~' and bar.notes:
            bar.notes[-1]['tie'] = True
    if waiting:
        bars[-1].after += waiting
    if not bars[-1].notes and bars[-1].clef is None and bars[-1].key is None and bars[-1].time is None \
            and not bars[-1].after and bars[-1].ending is None:
        bars.pop()
    return bars


def length(note: dict) -> Fraction:
    base = TYPE_LENGTH.get(note['type'], Fraction(1))
    total, add = base, base
    for _ in range(note['dots']):
        add /= 2
        total += add
    if note['tuplet']:
        actual, normal = note['tuplet'].split(':')
        total = total * Fraction(int(normal), int(actual))
    return total


def _time_value(t: str) -> tuple[int, int, str | None]:
    if t == 'C':
        return 4, 4, 'common'
    if t == 'C|':
        return 2, 2, 'cut'
    b, bt = t.split('/')
    return int(b), int(bt), None


def to_musicxml(staves_of_parts: list[list[list[Bar]]], names: list[str] | None = None,
                transpose: list[int] | None = None) -> bytes:
    """MusicXML of parts given as staves of bars: for each part, its
    staves, each a list of bars, in step with one another. The alteration
    of every pitch is worked out from the key and the accidentals printed
    earlier in its bar on its staff, as a reader of the page would."""
    root = ET.Element('score-partwise', version='4.0')
    plist = ET.SubElement(root, 'part-list')
    for p, staves in enumerate(staves_of_parts):
        sp = ET.SubElement(plist, 'score-part', id=f'P{p + 1}')
        ET.SubElement(sp, 'part-name').text = names[p] if names and p < len(names) else f'Part {p + 1}'
    # Endings are printed above the system's top staff and belong to every part.
    top = staves_of_parts[0][0] if staves_of_parts and staves_of_parts[0] else []
    for p, staves in enumerate(staves_of_parts):
        part = ET.SubElement(root, 'part', id=f'P{p + 1}')
        count = max(len(s) for s in staves)
        state = [{'clef': 'G2', 'key': 0, 'time': None, 'ties': {}} for _ in staves]
        ending = None  # the number of the ending open, if one is
        for i in range(count):
            m = ET.SubElement(part, 'measure', number=str(i + 1))
            here = top[i] if i < len(top) else Bar()
            forward = i > 0 and i - 1 < len(staves[0]) and staves[0][i - 1].barline in ('|:', ':|:')
            if forward or here.ending is not None:
                left = ET.SubElement(m, 'barline', location='left')
                if forward:
                    ET.SubElement(left, 'bar-style').text = 'heavy-light'
                if here.ending is not None:
                    import re
                    ending = ', '.join(re.findall(r'\d+', here.ending)) or '1'
                    ET.SubElement(left, 'ending', number=ending, type='start').text = here.ending
                if forward:
                    ET.SubElement(left, 'repeat', direction='forward')
            attrs = ET.SubElement(m, 'attributes')
            if i == 0:
                ET.SubElement(attrs, 'divisions').text = str(DIVISIONS)
                if transpose and p < len(transpose) and transpose[p]:
                    octaves = int(transpose[p] / 12)
                    tr = ET.Element('transpose')
                    ET.SubElement(tr, 'chromatic').text = str(transpose[p] - 12 * octaves)
                    if octaves:
                        ET.SubElement(tr, 'octave-change').text = str(octaves)
                    attrs.append(tr)
            first_bar = staves[0][i] if i < len(staves[0]) else Bar()
            key = first_bar.key
            if key is not None and (key != state[0]['key'] or i == 0):
                k = ET.SubElement(attrs, 'key')
                ET.SubElement(k, 'fifths').text = str(key)
            elif i == 0:
                k = ET.SubElement(attrs, 'key')
                ET.SubElement(k, 'fifths').text = '0'
            if first_bar.time and first_bar.time != state[0]['time']:
                b, bt, symbol = _time_value(first_bar.time)
                t = ET.SubElement(attrs, 'time', **({'symbol': symbol} if symbol else {}))
                ET.SubElement(t, 'beats').text = str(b)
                ET.SubElement(t, 'beat-type').text = str(bt)
            elif i == 0 and not any(bars and bars[0].time for bars in staves) and (meter := _meter(staves)):
                # A page that starts mid-work prints no time signature, but
                # its bars show the meter; it is written hidden, so a
                # whole-bar rest takes the bar's length.
                b, bt, _ = _time_value(meter)
                t = ET.SubElement(attrs, 'time', {'print-object': 'no'})
                ET.SubElement(t, 'beats').text = str(b)
                ET.SubElement(t, 'beat-type').text = str(bt)
                for st in state:
                    st['time'] = meter
            if i == 0 and len(staves) > 1:
                ET.SubElement(attrs, 'staves').text = str(len(staves))
            for s, bars in enumerate(staves):
                bar = bars[i] if i < len(bars) else Bar()
                if bar.key is not None:
                    state[s]['key'] = bar.key
                if bar.time:
                    state[s]['time'] = bar.time
                if bar.clef and (bar.clef != state[s]['clef'] or i == 0):
                    c = ET.SubElement(attrs, 'clef', number=str(s + 1))
                    sign, line, octave = _clef_parts(bar.clef)
                    ET.SubElement(c, 'sign').text = sign
                    if line:
                        ET.SubElement(c, 'line').text = line
                    if octave:
                        ET.SubElement(c, 'clef-octave-change').text = octave
                    state[s]['clef'] = bar.clef
                elif i == 0:
                    c = ET.SubElement(attrs, 'clef', number=str(s + 1))
                    ET.SubElement(c, 'sign').text = 'G'
                    ET.SubElement(c, 'line').text = '2'
            if (tr := attrs.find('transpose')) is not None:  # MusicXML puts it after the clefs
                attrs.remove(tr)
                attrs.append(tr)
            written = 0
            for s, bars in enumerate(staves):
                bar = bars[i] if i < len(bars) else Bar()
                if written:
                    back = ET.SubElement(m, 'backup')
                    ET.SubElement(back, 'duration').text = str(written)
                written = _write_notes(m, bar, state[s], s + 1, len(staves) > 1)
            stop = here.ending_end if ending is not None else None
            _write_barline(m, staves[0][i].barline if i < len(staves[0]) else '|',
                           fermata=i < len(staves[0]) and staves[0][i].fermata, ending_end=stop, ending=ending)
            if stop:
                ending = None
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def _meter(staves: list[list[Bar]]) -> str | None:
    """The time signature a part's bars add up to most often, for a part
    whose time signature is not printed: in quarters where they are whole,
    else in eighths."""
    import collections
    seen = collections.Counter()
    for bars in staves:
        for bar in bars:
            if any(n.get('measure') for n in bar.notes):
                continue
            ends: dict[int, Fraction] = {}
            for n in bar.notes:
                if not n.get('chord') and not n.get('grace'):
                    v = n.get('voice', 0)
                    ends[v] = ends.get(v, Fraction(0)) + length(n)
            if ends:
                seen[max(ends.values())] += 1
    if not seen:
        return None
    quarters = seen.most_common(1)[0][0]
    if quarters.denominator == 1:
        return f'{quarters.numerator}/4'
    if (2 * quarters).denominator == 1:
        return f'{2 * quarters.numerator // quarters.denominator}/8'
    return None


def _clef_parts(clef: str) -> tuple[str, str, str | None]:
    if clef == 'percussion':
        return 'percussion', '', None
    octave = None
    if clef.endswith(('+8', '-8')):
        octave = '1' if clef.endswith('+8') else '-1'
        clef = clef[:-2]
    return clef[0], clef[1:], octave


def _alter(step: str, key: int) -> int:
    if key > 0 and step in SHARPS[:key]:
        return 1
    if key < 0 and step in SHARPS[::-1][:-key]:
        return -1
    return 0


def _write_notes(m: ET.Element, bar: Bar, state: dict, staff: int, many: bool) -> int:
    """Write a staff's notes in a bar, voice after voice, and return where
    the last voice ends, in divisions."""
    key = state['key']
    _alterations(bar.notes, key, state)
    written = 0
    for v in sorted({n.get('voice', 0) for n in bar.notes}):
        if written:
            back = ET.SubElement(m, 'backup')
            ET.SubElement(back, 'duration').text = str(written)
        written = _write_voice(m, [n for n in bar.notes if n.get('voice', 0) == v], state, staff, many, v)
        if v == 0 and bar.after:
            from . import marks as M
            M.write_direction(m, bar.after, staff if many else None)
    if not bar.notes and bar.after:
        from . import marks as M
        M.write_direction(m, bar.after, staff if many else None)
    return written


def _alterations(notes: list[dict], key: int, state: dict):
    """Each note's alteration, worked out in the order the bar sounds: an
    accidental holds from where it is printed to the barline, in every
    voice of the staff, and a tie carries its note's alteration on."""
    onsets, at = [], {}
    for n in notes:
        v = n.get('voice', 0)
        here = at.get(v, Fraction(0))
        if n.get('chord'):
            here = onsets[-1] if onsets else here
        onsets.append(here)
        if not n.get('chord') and not n.get('grace') and not n.get('measure'):
            at[v] = here + length(n)
    altered: dict[str, int] = {}
    for _, k in sorted((o, k) for k, o in enumerate(onsets)):
        n = notes[k]
        if n['rest'] or n.get('space') or not n.get('pitch'):
            continue
        if n.get('accidental'):
            altered[n['pitch']] = ACCIDENTALS[n['accidental']]
        n['alter'] = altered.get(n['pitch'], _alter(n['pitch'][0], key))


def _write_voice(m, notes, state, staff, many, voice) -> int:
    from . import marks as M
    written = 0
    for n in notes:
        if n.get('before'):
            M.write_direction(m, n['before'], staff if many else None)
        if n.get('space'):
            f = ET.SubElement(m, 'forward')
            ET.SubElement(f, 'duration').text = str(int(length(n) * DIVISIONS))
            ET.SubElement(f, 'voice').text = str(voice + 1)
            if many:
                ET.SubElement(f, 'staff').text = str(staff)
            written += int(length(n) * DIVISIONS)
            continue
        el = ET.SubElement(m, 'note')
        if n.get('grace'):
            ET.SubElement(el, 'grace')
        if n.get('chord'):
            ET.SubElement(el, 'chord')
        if n['rest']:
            r = ET.SubElement(el, 'rest')
            if n['measure']:
                r.set('measure', 'yes')
        else:
            step, octave = n['pitch'][0], n['pitch'][1:]
            alter = n['alter']
            if not n.get('accidental') and (voice, n['pitch']) in state['ties']:
                alter = state['ties'][(voice, n['pitch'])]  # a tie carries its note's alteration
            p = ET.SubElement(el, 'pitch')
            ET.SubElement(p, 'step').text = step
            if alter:
                ET.SubElement(p, 'alter').text = str(alter)
            ET.SubElement(p, 'octave').text = octave
        if not n.get('grace'):
            if n['measure']:
                if state['time']:
                    b, bt, _ = _time_value(state['time'])
                    duration = Fraction(4 * b, bt)
                else:
                    duration = Fraction(4)
            else:
                duration = length(n)
            ET.SubElement(el, 'duration').text = str(int(duration * DIVISIONS))
            if not n.get('chord'):
                written += int(duration * DIVISIONS)
        # A tie belongs to its voice: it ends at that voice's next note of the pitch.
        stops = not n['rest'] and (voice, n['pitch']) in state['ties']
        if stops:
            del state['ties'][(voice, n['pitch'])]
            ET.SubElement(el, 'tie', type='stop')
        if n['tie']:
            ET.SubElement(el, 'tie', type='start')
        if not n['measure']:
            ET.SubElement(el, 'type').text = n['type']
            for _ in range(n['dots']):
                ET.SubElement(el, 'dot')
        if n.get('accidental') and not n['rest']:
            ET.SubElement(el, 'accidental').text = n['accidental']
        if n['tuplet']:
            actual, normal = n['tuplet'].split(':')
            tm = ET.SubElement(el, 'time-modification')
            ET.SubElement(tm, 'actual-notes').text = actual
            ET.SubElement(tm, 'normal-notes').text = normal
        if many:
            ET.SubElement(el, 'staff').text = str(staff)
        # MusicXML puts the voice after the duration and ties, before the type.
        v_el = ET.Element('voice')
        v_el.text = str(voice + 1)
        after = ('type', 'dot', 'accidental', 'time-modification', 'stem', 'notehead', 'staff', 'beam', 'notations')
        el.insert(next((k for k, c in enumerate(el) if c.tag in after), len(el)), v_el)
        if stops or n['tie']:
            notations = ET.SubElement(el, 'notations')
            if stops:
                ET.SubElement(notations, 'tied', type='stop')
            if n['tie']:
                ET.SubElement(notations, 'tied', type='start')
        if n['tie'] and not n['rest']:
            state['ties'][(voice, n['pitch'])] = alter
        if n.get('marks'):
            M.write_note_marks(el, n['marks'])
        for verse, text, syllabic, held in n.get('lyrics', []):  # set by lyrics.attach
            ly = ET.SubElement(el, 'lyric', number=str(verse))
            ET.SubElement(ly, 'syllabic').text = syllabic
            ET.SubElement(ly, 'text').text = text
            if held:
                ET.SubElement(ly, 'extend')
    return written


def _write_barline(m: ET.Element, barline: str, fermata: bool = False, ending_end: str | None = None,
                   ending: str | None = None):
    """The bar's right barline, with a fermata on it and an ending that
    stops there. A repeat that starts the next bar is written on that
    bar's left."""
    style = {'||': 'light-light', '|]': 'light-heavy', ':|': 'light-heavy', ':|:': 'light-heavy'}.get(barline)
    if not (style or fermata or ending_end):
        return
    b = ET.SubElement(m, 'barline', location='right')
    if style:
        ET.SubElement(b, 'bar-style').text = style
    if fermata:
        ET.SubElement(b, 'fermata')
    if ending_end:
        ET.SubElement(b, 'ending', number=ending or '1', type='stop' if ending_end == 'end|' else 'discontinue')
    if barline in (':|', ':|:'):
        ET.SubElement(b, 'repeat', direction='backward')


def _places(bar: Bar) -> list[tuple[int, Fraction, str | None]]:
    """Where each note of a bar stands: its voice, its onset, its pitch."""
    out, at, last = [], {}, Fraction(0)
    for n in bar.notes:
        v = n.get('voice', 0)
        here = last if n.get('chord') else at.get(v, Fraction(0))
        out.append((v, here, n.get('pitch')))
        if not n.get('chord'):
            last = here
            if not n.get('grace') and not n.get('measure'):
                at[v] = here + length(n)
    return out


def merge_marks(notes: list[Bar], marked: list[Bar]) -> list[Bar]:
    """The bars of a notes-only reading with the marks of a reading of the
    same staff that has them: each mark goes to the note standing where it
    stood (voice, onset and pitch), or at that onset in any voice, and a
    mark above or below the staff to the note it stood before. Where a
    misread rhythm has moved the onsets, a note of the same voice and pitch
    is taken in order. What has no such note is left out. The notes are the
    notes-only reading's."""
    for nb, mb in zip(notes, marked):
        here = _places(nb)
        by_place = {p: n for p, n in zip(here, nb.notes)}
        by_onset: dict[Fraction, dict] = {}
        for (v, onset, _), n in zip(here, nb.notes):
            if not n.get('space'):
                by_onset.setdefault(onset, n)
        # In order within a voice, where a misread rhythm has moved the onsets.
        queue: dict[tuple[int, str | None], list[dict]] = {}
        for (v, _, pitch), n in zip(here, nb.notes):
            if not n.get('space'):
                queue.setdefault((v, pitch), []).append(n)
        for (v, onset, pitch), m in zip(_places(mb), mb.notes):
            target = by_place.get((v, onset, pitch))
            if target is None and queue.get((v, pitch)):
                target = queue[(v, pitch)][0]
            if target is None:
                target = by_onset.get(onset)
            if target is not None and target in queue.get((v, pitch), []):
                queue[(v, pitch)].remove(target)
            if target is None:
                if m.get('before'):
                    nb.after += m['before']
                continue
            if m.get('marks'):
                target.setdefault('marks', [])
                target['marks'] += [t for t in m['marks'] if t not in target['marks']]
            if m.get('before'):
                target.setdefault('before', [])
                target['before'] += m['before']
        nb.after += mb.after
        nb.ending, nb.ending_end, nb.fermata = mb.ending, mb.ending_end, mb.fermata
    return notes
