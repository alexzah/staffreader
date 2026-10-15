"""Which instrument each staff of a system belongs to, from its label.

An orchestral score names each staff at the left of its first system,
often abbreviates the names on later systems, and leaves out the staves
of instruments with nothing to play. A staff's part is therefore read
from its label, not taken from its place. `instrument` turns a printed
label (Italian, German, English or French, in full or abbreviated) into
an instrument, the number of its player or section, and its
transposition; `assign` lines each system's staves up with the parts of
the score in order.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Printed names, lower case without accents or dots, to an instrument.
# Longer names come first, so 'kontrafagott' is not read as 'fagott'.
NAMES = [
    ('piccolo', 'piccolo'), ('flauto piccolo', 'piccolo'), ('kl fl', 'piccolo'), ('kleine flote', 'piccolo'),
    ('picc', 'piccolo'), ('ottavino', 'piccolo'),
    ('flauto', 'flute'), ('flauti', 'flute'), ('flote', 'flute'), ('floten', 'flute'), ('flute', 'flute'),
    ('fl', 'flute'), ('gr fl', 'flute'),
    ('englischhorn', 'english horn'), ('corno inglese', 'english horn'), ('cor ingl', 'english horn'),
    ('cor angl', 'english horn'), ('eh', 'english horn'),
    ('oboe', 'oboe'), ('oboi', 'oboe'), ('hautbois', 'oboe'), ('hob', 'oboe'), ('ob', 'oboe'),
    ('bassklarinette', 'bass clarinet'), ('clarinetto basso', 'bass clarinet'), ('bkl', 'bass clarinet'),
    ('clarinetto', 'clarinet'), ('clarinetti', 'clarinet'), ('klarinette', 'clarinet'), ('klarinetten', 'clarinet'),
    ('clarinet', 'clarinet'), ('klar', 'clarinet'), ('clar', 'clarinet'), ('cl', 'clarinet'), ('kl', 'clarinet'),
    ('kontrafagott', 'contrabassoon'), ('contrafagotto', 'contrabassoon'), ('k fag', 'contrabassoon'),
    ('kfag', 'contrabassoon'), ('cfg', 'contrabassoon'), ('c fag', 'contrabassoon'),
    ('fagotto', 'bassoon'), ('fagotti', 'bassoon'), ('fagott', 'bassoon'), ('fagotte', 'bassoon'),
    ('bassoon', 'bassoon'), ('basson', 'bassoon'), ('fag', 'bassoon'), ('fg', 'bassoon'), ('bsn', 'bassoon'),
    ('corno', 'horn'), ('corni', 'horn'), ('horn', 'horn'), ('horner', 'horn'), ('cor', 'horn'), ('hr', 'horn'),
    ('hn', 'horn'),
    ('tromba', 'trumpet'), ('trombe', 'trumpet'), ('trompete', 'trumpet'), ('trompeten', 'trumpet'),
    ('trumpet', 'trumpet'), ('trompette', 'trumpet'), ('trp', 'trumpet'), ('tpt', 'trumpet'), ('tr', 'trumpet'),
    ('clarino', 'trumpet'), ('clarini', 'trumpet'),
    ('trombone', 'trombone'), ('tromboni', 'trombone'), ('posaune', 'trombone'), ('posaunen', 'trombone'),
    ('pos', 'trombone'), ('trb', 'trombone'), ('tbn', 'trombone'), ('tromb', 'trombone'),
    ('basstuba', 'tuba'), ('tuba', 'tuba'),
    ('timpani', 'timpani'), ('pauken', 'timpani'), ('timbales', 'timpani'), ('timp', 'timpani'),
    ('pk', 'timpani'), ('timb', 'timpani'),
    ('arpa', 'harp'), ('harfe', 'harp'), ('harp', 'harp'), ('harpe', 'harp'), ('hp', 'harp'),
    ('violino', 'violin'), ('violini', 'violin'), ('violine', 'violin'), ('violinen', 'violin'),
    ('violin', 'violin'), ('violon', 'violin'), ('viol', 'violin'), ('vl', 'violin'), ('vn', 'violin'),
    ('vni', 'violin'), ('vln', 'violin'),
    ('viola', 'viola'), ('viole', 'viola'), ('bratsche', 'viola'), ('bratschen', 'viola'), ('br', 'viola'),
    ('vla', 'viola'), ('vle', 'viola'),
    ('violoncello', 'cello'), ('violoncelli', 'cello'), ('cello', 'cello'), ('vcl', 'cello'), ('vc', 'cello'),
    ('vlc', 'cello'),
    ('contrabasso', 'bass'), ('contrabassi', 'bass'), ('kontrabass', 'bass'), ('kontrabasse', 'bass'),
    ('contrebasse', 'bass'), ('cb', 'bass'), ('kb', 'bass'), ('k b', 'bass'), ('basso', 'bass'), ('bassi', 'bass'),
    ('soprano', 'soprano'), ('sopran', 'soprano'), ('tenore', 'tenor'), ('tenor', 'tenor'),
    ('pianoforte', 'piano'), ('piano', 'piano'), ('klavier', 'piano'), ('pfte', 'piano'),
    ('organo', 'organ'), ('orgel', 'organ'), ('organ', 'organ'),
]

# The key an instrument is pitched in, as printed ('in B', '(F)'), to the
# semitones its written notes sound: written C sounds that far away.
SOUNDING = {
    'clarinet': {'c': 0, 'b': -2, 'bb': -2, 'a': -3, 'es': 3, 'eb': 3, 'd': 2},
    'trumpet': {'c': 0, 'b': -2, 'bb': -2, 'a': -3, 'd': 2, 'es': 3, 'eb': 3, 'e': 4, 'f': 5},
    'horn': {'c': -12, 'd': -10, 'es': -9, 'eb': -9, 'e': -8, 'f': -7, 'g': -5, 'a': -3, 'b': -2, 'bb': -2},
}
FIXED = {'piccolo': 12, 'english horn': -7, 'bass clarinet': -14, 'contrabassoon': -12, 'bass': -12}


@dataclass(frozen=True)
class Instrument:
    name: str  # 'flute', 'horn', 'violin', ...
    number: str | None = None  # '1', '2', 'I', '1.2' where printed
    transpose: int = 0  # semitones from written to sounding pitch

    @property
    def key(self) -> str:
        return f'{self.name} {self.number}' if self.number else self.name


def _plain(label: str) -> str:
    text = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode().lower()
    text = re.sub(r'([a-z])(\d)|(\d)([a-z])', lambda m: ' '.join(g for g in m.groups() if g), text)  # 'vc1'
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', text).split())


def instrument(label: str | None) -> Instrument | None:
    """The instrument a printed staff label names, or None."""
    if not label:
        return None
    text = _plain(label)
    if not text:
        return None
    name = None
    for printed, inst in sorted(NAMES, key=lambda x: -len(x[0])):
        if re.search(rf'(^| ){re.escape(printed)}( |$)', text):
            name, rest = inst, re.sub(rf'(^| ){re.escape(printed)}( |$)', ' ', text, count=1).strip()
            break
    if name is None:
        # A name misread by a letter ('Vic' for 'Viol', 'Kla' for 'Klar'):
        # the printed name it is closest to, where it is close.
        import difflib
        words = text.split()
        best = (0.0, None, None)
        for k, w in enumerate(words):
            for span in (w, ' '.join(words[k:k + 2])):
                if len(span.replace(' ', '')) < 3:
                    continue
                for printed, inst in NAMES:
                    r = difflib.SequenceMatcher(None, span, printed).ratio()
                    if r > best[0]:
                        best = (r, inst, span)
        if best[0] < 0.85:
            return None
        name, rest = best[1], text.replace(best[2], ' ', 1).strip()
    number = None
    if (m := re.search(r'\b(i{1,3}|iv|[1-4])\b', rest)):
        number = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4'}.get(m.group(1), m.group(1))
    transpose = FIXED.get(name, 0)
    if name in SOUNDING:
        # 'in B', 'in F', '(C)': the key named after the instrument.
        keyed = re.search(r'(?:^| )(?:in )?(bb|eb|es|[a-g])(?: |$)', rest)
        if keyed and keyed.group(1) in SOUNDING[name]:
            transpose = SOUNDING[name][keyed.group(1)]
    return Instrument(name, number, transpose)


def assign(systems: list[list[str | None]]) -> tuple[list[Instrument], list[list[int | None]]]:
    """The parts of a score and, for each system, the part of each staff,
    from the labels of each system's staves (None where none was read).
    The first system's labels name the parts; a later system's labelled
    staves are matched by instrument in order, and an unlabelled staff
    takes the place between its labelled neighbours, or its place in the
    first system where the counts agree."""
    parts: list[Instrument] = []
    out: list[list[int | None]] = []
    for labels in systems:
        read = [instrument(l) for l in labels]
        if not parts:
            for k, inst in enumerate(read):
                parts.append(inst or Instrument(f'staff {k + 1}'))
            out.append(list(range(len(parts))))
            continue
        if len(labels) == len(parts) and all(r is None for r in read):
            out.append(list(range(len(parts))))
            continue
        before = {j for j in out[-1] if j is not None}
        out.append(_align(read, parts, before))
    return parts, out


def _align(read: list[Instrument | None], parts: list[Instrument], before: set[int] = frozenset()) -> list[int | None]:
    """Each staff's part, in order: the best monotone match of the
    system's staves to the score's parts, a staff matching a part of its
    instrument (and number, where both print one) and an unread staff
    matching any part, one the system before showed rather than one it
    left out."""
    n, m = len(read), len(parts)

    def score(i, j):
        r, p = read[i], parts[j]
        if r is None:
            return 0.3 if j in before else 0.1
        if r.name != p.name:
            return -1.0
        if r.number and p.number and r.number != p.number:
            return -0.5
        return 1.0

    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        best[i][0], move[i][0] = -9.0 * i, 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(best[i - 1][j - 1] + score(i - 1, j - 1), 3), (best[i][j - 1], 2), (best[i - 1][j] - 9.0, 1)]
            best[i][j], move[i][j] = max(options)
    pick: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if move[i][j] == 3:
            pick[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif move[i][j] == 2:
            j -= 1
        else:
            i -= 1
    return pick
