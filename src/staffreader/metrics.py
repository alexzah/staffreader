"""How a reading compares with the score we know is right.

Parts are matched first, by how alike their music is. Then each part's
measures are lined up with Needleman-Wunsch, so a missed or extra barline
costs one bar rather than shifting every bar after it. Inside each pair of
measures, events are matched at three levels:

- Core: written pitch and onset, what a listener hears;
- Rhythm: Core and the length;
- Full: Rhythm, the ties and the staff the note is on.

The headline is the bar-exact rate: the share of the score's bars a reader
got entirely right, which are the bars nobody has to correct.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

from .canonical import Event, Measure, Part, Score, printed

LEVELS = ('core', 'rhythm', 'full')
LAYERS = ('accidentals', 'marks', 'lyrics', 'slurs', 'dynamics')


def key(e: Event, level: str) -> tuple:
    """What an event must agree on to match at [level]."""
    base = (e.kind if e.kind != 'grace' else 'note', e.onset, e.pitch, e.kind == 'grace')
    if level == 'core':
        return base
    if level == 'rhythm':
        return base + (e.duration,)
    return base + (e.duration, e.tie_start, e.tie_stop, e.staff)


def _bag(m: Measure | None, level: str, kind: str = 'note') -> Counter:
    if m is None:
        return Counter()
    return Counter(key(e, level) for e in m.events if (e.kind == 'rest') == (kind == 'rest'))


def similarity(a: Measure, b: Measure) -> float:
    """How alike two measures' music is, from 0 to 1."""
    x, y = _bag(a, 'core') + _bag(a, 'core', 'rest'), _bag(b, 'core') + _bag(b, 'core', 'rest')
    union = sum((x | y).values())
    return sum((x & y).values()) / union if union else 1.0


def align(gt: list[Measure], pred: list[Measure], gap: float = -0.6) -> list[tuple[int | None, int | None]]:
    """Pairs of measure indices, (gt, pred), with None where one side has
    no measure. A measure read wrongly still pairs with its counterpart:
    matching two different measures (score down to -1) is preferred to
    two gaps (-1.2), so gaps are used only where the counts differ."""
    n, m = len(gt), len(pred)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0], back[i][0] = i * gap, 1
    for j in range(1, m + 1):
        score[0][j], back[0][j] = j * gap, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (score[i - 1][j - 1] + 2 * similarity(gt[i - 1], pred[j - 1]) - 1,
                       score[i - 1][j] + gap, score[i][j - 1] + gap)
            best = max(range(3), key=lambda k: options[k])
            score[i][j], back[i][j] = options[best], best
    pairs = []
    i, j = n, m
    while i or j:
        b = back[i][j]
        if b == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif b == 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


def match_parts(gt: Score, pred: Score) -> list[tuple[int, int | None]]:
    """Each score part paired with the reading's part most like it, each
    reading part used once."""
    def alike(a: Part, b: Part) -> float:
        pairs = align(a.measures, b.measures)
        return sum(similarity(a.measures[i], b.measures[j]) for i, j in pairs if i is not None and j is not None)
    candidates = sorted(((alike(a, b), i, j) for i, a in enumerate(gt.parts) for j, b in enumerate(pred.parts)),
                        reverse=True)
    used_gt, used_pred, out = set(), set(), {}
    for s, i, j in candidates:
        if i not in used_gt and j not in used_pred:
            used_gt.add(i)
            used_pred.add(j)
            out[i] = j
    return [(i, out.get(i)) for i in range(len(gt.parts))]


@dataclass
class Tally:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, gt: Counter, pred: Counter):
        both = sum((gt & pred).values())
        self.tp += both
        self.fn += sum(gt.values()) - both
        self.fp += sum(pred.values()) - both

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def merge(self, other: 'Tally'):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn


@dataclass
class BarResult:
    part: int
    gt: int | None
    pred: int | None
    exact_core: bool
    exact_full: bool
    notes_gt: int
    notes_missed: int
    notes_extra: int
    valid: bool | None  # does the read bar add up? None where nothing was read


@dataclass
class Report:
    notes: dict[str, Tally] = field(default_factory=lambda: {lv: Tally() for lv in LEVELS})
    rests: Tally = field(default_factory=Tally)
    layers: dict[str, Tally] = field(default_factory=lambda: {ly: Tally() for ly in LAYERS})
    bars: list[BarResult] = field(default_factory=list)
    attributes: Tally = field(default_factory=Tally)  # clef/key/time in force, per measure
    structure: Tally = field(default_factory=Tally)  # repeats, endings, double bars
    parts_gt: int = 0
    parts_read: int = 0
    parts_unmatched: int = 0
    measures_missed: int = 0
    measures_extra: int = 0
    errors: Counter = field(default_factory=Counter)  # each mistake, by kind
    families: dict[str, Tally] = field(default_factory=dict)  # every mark, by family (see family)

    def summary(self, masks: frozenset[str] = frozenset()) -> dict:
        gt_bars = [b for b in self.bars if b.gt is not None]
        read = [b for b in self.bars if b.pred is not None and b.valid is not None]
        out = {f'note_f1_{lv}': round(self.notes[lv].f1, 4) for lv in LEVELS}
        out |= {
            'note_precision_core': round(self.notes['core'].precision, 4),
            'note_recall_core': round(self.notes['core'].recall, 4),
            'rest_f1': round(self.rests.f1, 4),
            'bar_exact_core': round(sum(b.exact_core for b in gt_bars) / len(gt_bars), 4) if gt_bars else 1.0,
            'bar_exact_full': round(sum(b.exact_full for b in gt_bars) / len(gt_bars), 4) if gt_bars else 1.0,
            'bar_valid': round(sum(bool(b.valid) for b in read) / len(read), 4) if read else None,
            'attributes': round(self.attributes.f1, 4),
            'structure': round(self.structure.f1, 4),
            'parts': f'{self.parts_gt - self.parts_unmatched}/{self.parts_gt} matched, {self.parts_read} read',
            'measures_missed': self.measures_missed,
            'measures_extra': self.measures_extra,
            'bars': len(gt_bars),
            'notes': self.notes['core'].tp + self.notes['core'].fn,
            'errors': dict(self.errors.most_common()),
        }
        for ly in LAYERS:
            if ly not in masks:
                out[f'{ly}_f1'] = round(self.layers[ly].f1, 4)
        out['marks'] = {f: {'f1': round(t.f1, 4), 'printed': t.tp + t.fn, 'read': t.tp + t.fp}
                        for f, t in sorted(self.families.items()) if f not in masks}
        return out


def _attributes(m: Measure | None) -> Counter:
    if m is None:
        return Counter()
    items = [f'clef{s}={c}' for s, c in m.clefs.items()] + [f'key={m.key}', f'time={m.time}']
    return Counter(items)


def _layer(m: Measure | None, layer: str, matched: list[tuple[Event, Event]] | None = None) -> Counter:
    if m is None:
        return Counter()
    if layer == 'slurs':
        return Counter({'slur': m.slurs}) if m.slurs else Counter()
    if layer == 'dynamics':
        return Counter(m.dynamics)
    attr = {'accidentals': 'accidental', 'marks': 'marks', 'lyrics': 'lyric'}[layer]
    out = Counter()
    for e in m.events:
        value = getattr(e, attr)
        if value:
            out[(e.onset, e.pitch, value)] += 1
    return out


def family(label: str) -> str:
    """The family a mark's label belongs to, as the inventory groups them,
    so each is scored on its own."""
    from .canonical import ARTICULATIONS
    if label.startswith('slur-'):
        return 'slurs'
    if label in ARTICULATIONS or label == 'articulation:other':
        return 'articulations'
    if label.startswith('tremolo:'):
        return 'tremolos'
    if label in ('fermata', 'barline-fermata'):
        return 'fermatas'
    if label in ('arpeggiate', 'non-arpeggiate'):
        return 'arpeggios'
    if label.startswith('fingering:'):
        return 'fingering'
    if label.startswith('head:'):
        return 'noteheads'
    if label == 'cautionary':
        return 'cautionary accidentals'
    if label == 'glissando':
        return 'glissandos'
    if label in ('up-bow', 'down-bow', 'harmonic', 'open-string', 'stopped', 'snap-pizzicato', 'thumb-position',
                 'double-tongue', 'triple-tongue', 'heel', 'toe'):
        return 'bowings and technique'
    if label.startswith('dyn:'):
        return 'dynamics'
    if label.startswith('wedge:'):
        return 'hairpins'
    if label.startswith(('words:', 'dashes:')):
        return 'words'
    if label.startswith('metronome:'):
        return 'tempo marks'
    if label.startswith('octave:'):
        return 'octave lines'
    if label.startswith('pedal:'):
        return 'pedal'
    if label.startswith('rehearsal:'):
        return 'rehearsal marks'
    if label in ('segno', 'coda', 'barline-segno', 'barline-coda') or label.startswith('jump:'):
        return 'jumps'
    if label.startswith('multirest:'):
        return 'multi-bar rests'
    return 'ornaments'  # trills, mordents, turns, wavy lines, accidentals on ornaments


# Marks drawn once for a stem, which a file may put on one note of a
# chord or on each; the rest belong to a note (a fingering, a notehead).
PER_STEM = {'slurs', 'articulations', 'tremolos', 'fermatas', 'arpeggios', 'ornaments', 'bowings and technique',
            'glissandos', 'dynamics'}


def _families(m: Measure | None) -> dict[str, Counter]:
    """Every mark in a measure, by family: a note's marks where the note
    is (staff, onset, pitch), the rest where they fall in the bar."""
    out: dict[str, Counter] = {}
    if m is None:
        return out
    for e in m.events:
        for label in e.marks:
            f = family(label)
            if f in PER_STEM:  # one per stem, however a chord's notes carry it
                out.setdefault(f, Counter())[(e.staff, e.onset, label)] = 1
            else:
                out.setdefault(f, Counter())[(e.staff, e.onset, e.pitch, label)] += 1
        for ly in e.lyrics:
            out.setdefault('lyrics', Counter())[(e.staff, e.onset, ly)] += 1
    for onset, label in m.directions:
        out.setdefault(family(label), Counter())[(onset, label)] += 1
    for onset, pitch in m.cues:
        out.setdefault('cue notes', Counter())[(onset, pitch)] += 1
    for b in m.barlines:
        if b.startswith('ending-'):
            out.setdefault('endings', Counter())[b] += 1
    return out


def classify(mg: Measure | None, mp: Measure | None) -> Counter:
    """Each mistake in one pair of measures, by kind: a note missed or
    added, a wrong pitch, a wrong accidental on the right note, a wrong
    length, a grace note read as a full one or the other way, a rest missed
    or added, and the clefs, key, time and repeats printed in the bar."""
    out = Counter()
    gn = [e for e in (mg.events if mg else []) if e.kind != 'rest']
    pn = [e for e in (mp.events if mp else []) if e.kind != 'rest']
    left_g, left_p = list(gn), list(pn)
    for e in list(left_g):  # right pitch and onset
        for f in left_p:
            if (e.onset, e.pitch) == (f.onset, f.pitch):
                if e.kind != f.kind:
                    out['grace vs full note'] += 1
                elif e.duration != f.duration:
                    out['wrong length'] += 1
                left_g.remove(e)
                left_p.remove(f)
                break
    for e in list(left_g):  # same onset, another pitch
        for f in left_p:
            if e.onset == f.onset:
                same_line = e.pitch[0] == f.pitch[0] and e.pitch[-1] == f.pitch[-1]
                out['wrong accidental' if same_line else 'wrong pitch'] += 1
                left_g.remove(e)
                left_p.remove(f)
                break
    out['missed note'] += len(left_g)
    out['extra note'] += len(left_p)
    gr = Counter((e.onset, e.duration) for e in (mg.events if mg else []) if e.kind == 'rest')
    pr = Counter((e.onset, e.duration) for e in (mp.events if mp else []) if e.kind == 'rest')
    out['missed rest'] += sum((gr - pr).values())
    out['extra rest'] += sum((pr - gr).values())
    if mg and mp:
        for name in ('clef', 'key', 'time'):
            g = {x for x in printed(mg) if x.startswith(name)}
            q = {x for x in printed(mp) if x.startswith(name)}
            if g != q:
                out[f'{name} printed wrong or missed'] += 1
        if mg.barlines != mp.barlines:
            out['repeat or ending'] += len(mg.barlines ^ mp.barlines)
    return +out


def valid(measures: list[Measure], j: int) -> bool:
    """Whether read measure [j] adds up to its time signature. A pickup at
    the start, a short last bar, and a bar split in two by a repeat or a
    section's end, whose halves add up together, are allowed."""
    m = measures[j]
    if m.nominal is None:
        return True
    length = m.length()
    if length == m.nominal:
        return True
    if length > m.nominal:
        return False
    if j == 0 or j == len(measures) - 1 or m.implicit:
        return True
    return any(0 <= k < len(measures) and measures[k].length() + length == m.nominal for k in (j - 1, j + 1))


def compare(gt: Score, pred: Score) -> Report:
    report = Report(parts_gt=len(gt.parts), parts_read=len(pred.parts))
    for gi, pi in match_parts(gt, pred):
        g = gt.parts[gi]
        p = pred.parts[pi] if pi is not None else None
        if p is None:
            report.parts_unmatched += 1
        elif g.transpose or p.transpose:
            report.families.setdefault('transposition', Tally()).add(Counter({g.transpose: 1}), Counter({p.transpose: 1}))
        pairs = align(g.measures, p.measures) if p else [(i, None) for i in range(len(g.measures))]
        for i, j in pairs:
            mg = g.measures[i] if i is not None else None
            mp = p.measures[j] if (p and j is not None) else None
            if i is None:
                report.measures_extra += 1
            if j is None:
                report.measures_missed += 1
            report.errors += classify(mg, mp)
            for lv in LEVELS:
                report.notes[lv].add(_bag(mg, lv), _bag(mp, lv))
            report.rests.add(_bag(mg, 'rhythm', 'rest'), _bag(mp, 'rhythm', 'rest'))
            for ly in LAYERS:
                report.layers[ly].add(_layer(mg, ly), _layer(mp, ly))
            report.attributes.add(_attributes(mg), _attributes(mp))
            fg, fp = _families(mg), _families(mp)
            for f in set(fg) | set(fp):
                report.families.setdefault(f, Tally()).add(fg.get(f, Counter()), fp.get(f, Counter()))
            report.structure.add(Counter(mg.barlines) if mg else Counter(), Counter(mp.barlines) if mp else Counter())
            core_g = _bag(mg, 'core') + _bag(mg, 'rhythm', 'rest')
            core_p = _bag(mp, 'core') + _bag(mp, 'rhythm', 'rest')
            full_g = _bag(mg, 'full') + _bag(mg, 'full', 'rest')
            full_p = _bag(mp, 'full') + _bag(mp, 'full', 'rest')
            notes_g, notes_p = _bag(mg, 'core'), _bag(mp, 'core')
            both = sum((notes_g & notes_p).values())
            report.bars.append(BarResult(
                part=gi, gt=i, pred=j,
                exact_core=mg is not None and mp is not None and core_g == core_p,
                exact_full=(mg is not None and mp is not None and full_g == full_p
                            and printed(mg) == printed(mp) and mg.barlines == mp.barlines),
                notes_gt=sum(notes_g.values()), notes_missed=sum(notes_g.values()) - both,
                notes_extra=sum(notes_p.values()) - both,
                valid=valid(p.measures, j) if mp is not None else None))
    return report
