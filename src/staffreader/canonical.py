"""A score read into the form the harness compares: for each part, its
measures in printed order, repeats not unfolded, and in each measure the
events as printed.

An event is a note, a rest or a grace note, with its onset in the measure
as an exact fraction of a quarter note, its written pitch, its length and
its ties. Voices are ignored, because two editions or two readers may
number them differently and the music is the same. Anything the score
hides (print-object="no") is left out, because no reader can see it.

Each measure also carries what is in force there (clef of each staff,
key, time), what it changes, its barlines (repeats and endings), and the
secondary layers anchored to it: printed accidentals, slurs, dynamics,
articulations and ornaments, lyrics and words.
"""

from __future__ import annotations

import io
import re
import pathlib
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction

STEPS = 'CDEFGAB'


@dataclass(frozen=True)
class Event:
    kind: str  # 'note', 'rest' or 'grace'
    staff: int
    onset: Fraction
    duration: Fraction
    pitch: str | None = None  # written, such as 'F#4'; None for a rest
    tie_start: bool = False
    tie_stop: bool = False
    accidental: str | None = None  # the printed accidental, if any
    marks: tuple[str, ...] = ()  # every mark on the note, sorted (see _note_marks)
    lyric: str | None = None  # verse 1's syllable
    lyrics: tuple[str, ...] = ()  # every verse: 'verse:text', with '-' for a hyphen after, '_' for an extender


@dataclass
class Measure:
    index: int  # position in the part, from 0
    number: str
    implicit: bool
    events: list[Event] = field(default_factory=list)
    clefs: dict[int, str] = field(default_factory=dict)  # in force, by staff
    key: int = 0  # fifths in force
    time: tuple[int, int] | None = None  # in force
    changes: frozenset[str] = frozenset()  # attributes changed here
    barlines: frozenset[str] = frozenset()  # 'repeat-forward', 'ending-1-start', ...
    slurs: int = 0  # slur starts
    dynamics: tuple[str, ...] = ()
    words: tuple[str, ...] = ()
    # Marks printed above or below the staff, where they fall in the bar:
    # (onset, label), such as (0, 'dyn:p') or (2, 'wedge:crescendo').
    directions: tuple[tuple[Fraction, str], ...] = ()
    cues: tuple[tuple[Fraction, str], ...] = ()  # cue notes (small, not played): (onset, pitch or 'R')

    @property
    def nominal(self) -> Fraction | None:
        """The length the time signature gives the measure, in quarters."""
        if self.time is None:
            return None
        beats, beat_type = self.time
        return Fraction(4 * beats, beat_type)

    def length(self) -> Fraction:
        """How long the measure's music runs, the longest staff's."""
        ends = [e.onset + e.duration for e in self.events if e.kind != 'grace']
        return max(ends, default=Fraction(0))


@dataclass
class Part:
    id: str
    name: str
    staves: int
    measures: list[Measure]
    transpose: int = 0  # semitones from written to sounding pitch, for a transposing instrument


@dataclass
class Score:
    parts: list[Part]

    def excerpt(self, first: int, last: int) -> 'Score':
        """Measures [first, last] of every part, counted from 1 by
        position, as a case names its bars."""
        return Score([Part(p.id, p.name, p.staves,
                           [_reindex(m, i) for i, m in enumerate(p.measures[first - 1:last])], p.transpose)
                      for p in self.parts])


def _reindex(m: Measure, i: int) -> Measure:
    """[m] as measure [i] of an excerpt. An excerpt's first measure prints
    every clef and the key, as the first bar of a system does; it prints
    the time only where the score does, since a page that starts mid-work
    prints none."""
    out = Measure(**{**m.__dict__})
    out.index = i
    if i == 0:
        out.changes = frozenset({f'clef{s}' for s in m.clefs} | {'key'} | ({'time'} & m.changes))
    return out


def printed(m: Measure) -> frozenset[str]:
    """The clefs, key and time printed in [m], with their values."""
    values = {f'clef{s}': c for s, c in m.clefs.items()} | {'key': m.key, 'time': m.time}
    return frozenset(f'{name}={values.get(name)}' for name in m.changes)


def read(path: str | pathlib.Path) -> Score:
    """The canonical form of a MusicXML file, plain or compressed."""
    return parse(load(path))


# The Hauptstimme corpus marks its analysis of OpenScore's orchestral
# scores as red lyrics: a letter for each theme ("a", "b'"), and notes such
# as "(dev)" or "cad.". They are not printed in the edition and are never
# sung, which is never red, so they are taken out wherever a score is read.
_ANALYSIS = re.compile(rb'<lyric\b[^>]*color="#ff0000"[^>]*>.*?</lyric>', re.IGNORECASE | re.DOTALL)


def load(path: str | pathlib.Path) -> bytes:
    path = pathlib.Path(path)
    data = path.read_bytes()
    if path.suffix == '.mxl' or data[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            container = ET.fromstring(z.read('META-INF/container.xml'))
            rootfile = next(el for el in container.iter() if el.tag.endswith('rootfile'))
            data = z.read(rootfile.get('full-path'))
    if b'FF0000' in data.upper():
        data = _ANALYSIS.sub(b'', data)
    return data


def parse(xml: bytes | str) -> Score:
    root = ET.fromstring(xml)
    for el in root.iter():  # some writers namespace the elements
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]
    if root.tag != 'score-partwise':
        raise ValueError(f'expected score-partwise, not {root.tag}')
    names = {sp.get('id'): (sp.findtext('part-name') or '').strip()
             for sp in root.iter('score-part')}
    return Score([_part(p, names.get(p.get('id'), '')) for p in root.findall('part')])


def _part(part: ET.Element, name: str) -> Part:
    divisions = 1
    clefs: dict[int, str] = {1: 'G2'}
    key = 0
    time: tuple[int, int] | None = None
    staves = 1
    transpose = 0
    measures = []
    for i, m in enumerate(part.findall('measure')):
        events: list[Event] = []
        changes: set[str] = set()
        barlines: set[str] = set()
        slurs = 0
        dynamics: list[str] = []
        words: list[str] = []
        directions: list[tuple[Fraction, str]] = []
        cues: list[tuple[Fraction, str]] = []
        pos = Fraction(0)
        last_onset = Fraction(0)
        for el in m:
            if el.tag == 'attributes':
                if (n := el.findtext('measure-style/multiple-rest')) and n.strip().isdigit():
                    directions.append((Fraction(0), None, f'multirest:{int(n)}'))
                if (d := el.findtext('divisions')) is not None:
                    divisions = int(float(d))
                if (tr := el.find('transpose')) is not None and not transpose:
                    transpose = int(tr.findtext('chromatic', '0')) + 12 * int(tr.findtext('octave-change', '0'))
                for ms in el.findall('measure-style'):
                    if (mr := ms.findtext('multiple-rest')):
                        directions.append((pos, None, f'multirest:{mr}'))
                if (s := el.findtext('staves')) is not None:
                    staves = max(staves, int(s))
                for c in el.findall('clef'):
                    staff = int(c.get('number', '1'))
                    clef = f"{c.findtext('sign', 'G')}{c.findtext('line', '')}"
                    octave = c.findtext('clef-octave-change')
                    if octave and octave != '0':
                        clef += f'{int(octave):+d}'
                    if clefs.get(staff) != clef or i == 0:
                        changes.add(f'clef{staff}')
                    clefs[staff] = clef
                for k in el.findall('key'):
                    if (f := k.findtext('fifths')) is not None:
                        if int(f) != key or i == 0:
                            changes.add('key')
                        key = int(f)
                for t in el.findall('time'):
                    beats, beat_type = t.findtext('beats'), t.findtext('beat-type')
                    if beats and beat_type and '+' not in beats:
                        new = (int(beats), int(beat_type))
                        if (new != time or i == 0) and t.get('print-object') != 'no':
                            changes.add('time')
                        time = new
            elif el.tag == 'backup':
                pos -= Fraction(int(float(el.findtext('duration', '0'))), divisions)
            elif el.tag == 'forward':
                pos += Fraction(int(float(el.findtext('duration', '0'))), divisions)
            elif el.tag == 'note':
                grace = el.find('grace') is not None
                chord = el.find('chord') is not None
                duration = Fraction(0) if grace else Fraction(int(float(el.findtext('duration', '0'))), divisions)
                onset = last_onset if chord else pos
                if not chord and not grace:
                    pos += duration
                if not chord:
                    last_onset = onset
                # A note may carry its notations in more than one element.
                notations = ET.Element('notations')
                for n in el.findall('notations'):
                    notations.extend(list(n))
                if el.find('cue') is not None:  # small notes to follow by, not played
                    if el.get('print-object') != 'no':
                        p = el.find('pitch')
                        cues.append((onset, 'R' if p is None else
                                     f"{p.findtext('step')}{_alter(float(p.findtext('alter', '0')))}{p.findtext('octave')}"))
                    continue
                slurs += sum(1 for s in notations.findall('slur') if s.get('type') == 'start')
                if el.get('print-object') == 'no':
                    continue
                staff = int(el.findtext('staff', '1'))
                rest = el.find('rest') is not None
                pitch = None
                if not rest:
                    p = el.find('pitch')
                    if p is None:
                        p = el.find('unpitched')
                        step, octave, alter = (p.findtext('display-step', 'C'), p.findtext('display-octave', '4'), 0) \
                            if p is not None else ('C', '4', 0)
                    else:
                        step, octave = p.findtext('step'), p.findtext('octave')
                        alter = float(p.findtext('alter', '0'))
                    pitch = f"{step}{_alter(alter)}{octave}"
                ties = {t.get('type') for t in el.findall('tie')}
                if notations is not None:
                    ties |= {t.get('type') for t in notations.findall('tied')}
                marks = _note_marks(el, notations)
                lyric, lyrics = None, []
                for ly in el.findall('lyric'):
                    text = (ly.findtext('text') or '').strip()
                    if not text:
                        continue
                    if ly.get('number', '1') == '1':
                        lyric = text
                    after = ('-' if ly.findtext('syllabic') in ('begin', 'middle') else '') + \
                            ('_' if ly.find('extend') is not None else '')
                    lyrics.append(f"{ly.get('number', '1')}:{text}{after}")
                events.append(Event(
                    kind='rest' if rest else ('grace' if grace else 'note'),
                    staff=staff, onset=onset, duration=duration, pitch=pitch,
                    tie_start='start' in ties, tie_stop='stop' in ties,
                    accidental=(el.findtext('accidental') or None) if not rest else None,
                    marks=tuple(sorted(marks)), lyric=lyric, lyrics=tuple(sorted(lyrics))))
            elif el.tag == 'direction':
                for d in el.iter('dynamics'):
                    dynamics += [c.tag for c in d]
                for w in el.iter('words'):
                    if w.text and w.text.strip():
                        words.append(w.text.strip())
                at = pos + Fraction(int(float(el.findtext('offset', '0'))), divisions)
                where = int(el.findtext('staff', '1'))
                directions += [(at, where, label) for label in _direction_labels(el)]
            elif el.tag == 'sound':
                directions += [(pos, 1, label) for label in _jumps(el)]
            elif el.tag == 'harmony' and el.get('print-object') != 'no' and el.find('root') is not None:
                at = pos + Fraction(int(float(el.findtext('offset', '0'))), divisions)
                directions.append((at, int(el.findtext('staff', '1')), _chord_label(el)))
            elif el.tag == 'figured-bass' and el.get('print-object') != 'no' and \
                    any(len(f) for f in el.findall('figure')):  # an empty one prints nothing
                directions.append((pos, None, _figures_label(el)))
            elif el.tag == 'barline':
                side = el.get('location', 'right')
                if (r := el.find('repeat')) is not None:
                    barlines.add(f"repeat-{r.get('direction')}")
                for e in el.findall('ending'):
                    for n in (e.get('number') or '').replace(' ', '').split(','):
                        if n:
                            barlines.add(f"ending-{n}-{e.get('type')}")
                for mark in ('segno', 'coda', 'fermata'):
                    if el.find(mark) is not None:
                        directions.append((Fraction(0) if side == 'left' else pos, None, f'barline-{mark}'))
                style = el.findtext('bar-style')
                if style in ('light-light', 'light-heavy', 'heavy-light', 'heavy-heavy') and side == 'right' \
                        and el.find('repeat') is None:
                    barlines.add(f'bar-{style}')
        measures.append(Measure(
            index=i, number=m.get('number', str(i + 1)), implicit=m.get('implicit') == 'yes',
            events=sorted(events, key=_order), clefs=dict(clefs), key=key, time=time,
            changes=frozenset(changes), barlines=frozenset(barlines), slurs=slurs,
            dynamics=tuple(sorted(dynamics)), words=tuple(words),
            directions=_snapped(directions, events), cues=tuple(sorted(cues))))
    return Part(part.get('id', ''), name, staves, measures, transpose)


ARTICULATIONS = {'staccato', 'staccatissimo', 'spiccato', 'accent', 'strong-accent', 'tenuto', 'detached-legato',
                 'soft-accent', 'stress', 'unstress', 'breath-mark', 'caesura', 'scoop', 'plop', 'doit', 'falloff'}


def _snapped(directions: list[tuple], events: list[Event]) -> tuple[tuple[Fraction, str], ...]:
    """Each mark above or below a staff where a reader can place it: at
    the first note or rest that staff shows at or after it, or at the
    barline if none follows. A hairpin that ends in the middle of a long
    note ends at the next note, or at the bar's end."""
    out = []
    for at, staff, label in directions:
        if staff is not None:
            mine = [e for e in events if e.staff == staff]
            later = [e.onset for e in mine if e.onset >= at]
            at = min(later, default=max((e.onset + e.duration for e in mine if e.kind != 'grace'), default=at))
        out.append((at, label))
    return tuple(sorted(out))


def _note_marks(note: ET.Element, notations: ET.Element | None) -> list[str]:
    """Every mark printed on a note, as a label: articulations, ornaments
    and tremolos, fermatas and arpeggios, bowings and fingerings, slur
    ends, glissandos, a notehead of another shape, and a cautionary or
    editorial accidental."""
    out = []
    head = note.findtext('notehead')
    if head and head not in ('normal', 'none'):
        out.append(f'head:{head}')
    acc = note.find('accidental')
    # Printed in parentheses or brackets: a courtesy accidental marked only
    # cautionary="yes" prints like any other.
    if acc is not None and 'yes' in (acc.get('parentheses'), acc.get('bracket')):
        out.append('cautionary')
    if notations is None:
        return out
    for child in notations:
        if child.tag == 'articulations':
            out += [g.tag if g.tag in ARTICULATIONS else 'articulation:other' for g in child]
        elif child.tag == 'ornaments':
            for g in child:
                if g.tag == 'tremolo':
                    kind = g.get('type', 'single')
                    if kind != 'stop':
                        out.append(f"tremolo:{kind}-{(g.text or '3').strip()}")
                elif g.tag == 'wavy-line':
                    if g.get('type') == 'start':
                        out.append('wavy-line')
                elif g.tag == 'accidental-mark':
                    out.append(f"accmark:{(g.text or '').strip()}")
                else:
                    out.append(g.tag)
        elif child.tag == 'technical':
            for g in child:
                if g.tag == 'fingering':
                    out.append(f"fingering:{(g.text or '').strip()}")
                elif g.tag in ('string', 'fret'):
                    out.append(f"{g.tag}:{(g.text or '').strip()}")
                elif g.tag in ('pluck', 'hammer-on', 'pull-off', 'tap'):
                    continue  # guitar marks outside the inventory
                else:
                    out.append(g.tag)
        elif child.tag == 'slur' and child.get('type') in ('start', 'stop'):
            out.append(f"slur-{child.get('type')}")
        elif child.tag in ('glissando', 'slide') and child.get('type') == 'start':
            out.append('glissando')
        elif child.tag in ('fermata', 'arpeggiate', 'non-arpeggiate'):
            out.append(child.tag)
        elif child.tag == 'dynamics':
            out += [f'dyn:{g.tag}' for g in child]
    return out


def _direction_labels(direction: ET.Element) -> list[str]:
    """What a direction prints, as labels: dynamics, hairpins, words,
    metronome marks, octave lines, pedal marks, rehearsal marks, segno and
    coda, dashed lines, and any jump its sound carries."""
    out = []
    if direction.get('print-object') == 'no':
        return out
    for dt in direction.findall('direction-type'):
        for g in dt:
            if g.get('print-object') == 'no':
                continue
            if g.tag == 'dynamics':
                out += [f'dyn:{d.tag}' if d.tag != 'other-dynamics' else f"dyn:{(d.text or '').strip()}" for d in g]
            elif g.tag == 'wedge' and g.get('type') in ('crescendo', 'diminuendo', 'stop'):
                out.append(f"wedge:{g.get('type')}")
            elif g.tag == 'words' and g.text and g.text.strip():
                from .marks import readable
                if readable(g.text):  # a music font's symbols are not words
                    out.append(f"words:{' '.join(g.text.split())}")
            elif g.tag == 'metronome':
                unit = (g.findtext('beat-unit') or '') + '.' * len(g.findall('beat-unit-dot'))
                out.append(f"metronome:{unit}={(g.findtext('per-minute') or '').strip()}")
            elif g.tag == 'octave-shift' and g.get('type') in ('up', 'down', 'stop'):
                out.append(f"octave:{g.get('type')}" + (f"-{g.get('size', '8')}" if g.get('type') != 'stop' else ''))
            elif g.tag == 'pedal' and g.get('type') in ('start', 'stop', 'change'):
                out.append(f"pedal:{g.get('type')}")
            elif g.tag == 'rehearsal' and g.text and g.text.strip():
                out.append(f'rehearsal:{g.text.strip()}')
            elif g.tag in ('segno', 'coda'):
                out.append(g.tag)
            elif g.tag == 'dashes' and g.get('type') in ('start', 'stop'):
                out.append(f"dashes:{g.get('type')}")
    if (sound := direction.find('sound')) is not None:
        out += _jumps(sound)
    return out


def _chord_label(h: ET.Element) -> str:
    """A chord symbol by what it means: root, kind, degrees and bass."""
    def step(el, tag):
        return f"{el.findtext(tag + '-step')}{_alter(float(el.findtext(tag + '-alter') or 0))}" if el is not None else ''
    degrees = ''.join(f"({d.findtext('degree-type')}{_alter(float(d.findtext('degree-alter') or 0))}"
                      f"{(d.findtext('degree-value') or '').strip()})" for d in h.findall('degree'))
    bass = h.find('bass')
    return f"chord:{step(h.find('root'), 'root')} {(h.findtext('kind') or 'major').strip()}{degrees}" + \
        (f"/{step(bass, 'bass')}" if bass is not None else '')


def _figures_label(fb: ET.Element) -> str:
    """A figured bass by its figures, top to bottom."""
    return 'figures:' + ','.join(f"{f.findtext('prefix') or ''}{(f.findtext('figure-number') or '').strip()}"
                                 f"{f.findtext('suffix') or ''}{'_' if f.find('extend') is not None else ''}"
                                 for f in fb.findall('figure'))


def _jumps(sound: ET.Element) -> list[str]:
    """The jumps in the order of play a sound element carries."""
    return [f'jump:{a}' for a in ('dacapo', 'dalsegno', 'fine', 'tocoda') if sound.get(a)]


def _alter(alter: float) -> str:
    return {-2: 'bb', -1: 'b', 0: '', 1: '#', 2: '##'}.get(round(alter), '')


def _order(e: Event):
    return (e.staff, e.onset, e.kind != 'grace', e.pitch or '', e.duration)


def midi(pitch: str) -> int:
    """The MIDI number of a written pitch such as 'F#4', to compare pitches
    spelled two ways."""
    step, rest = pitch[0], pitch[1:]
    alter = 0
    while rest and rest[0] in '#b':
        alter += 1 if rest[0] == '#' else -1
        rest = rest[1:]
    return 12 * (int(rest) + 1) + [0, 2, 4, 5, 7, 9, 11][STEPS.index(step)] + alter
