"""The right layout of a page, and how close a found layout comes to it.

For a page MuseScore engraved, the answer is exact: its SVG export marks
every staff line and barline (classes StaffLines and BarLine), in units
of 1/1200 inch, a quarter of a pixel at 300 dpi. Staves are five lines
sharing their ends; consecutive staves are one system when their
barlines fall at the same places.

For a real scan the answer is what a person counted on the page and wrote
in the case: how many staves each system has and how many bars.
"""

from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .layout import Layout, Staff, System

SVG_PER_PIXEL = 4.0  # 1200 units an inch, 300 pixels an inch


def from_svg(path: pathlib.Path, scale: float = SVG_PER_PIXEL) -> Layout:
    root = ET.parse(path).getroot()
    lines, bars, symbols = [], [], []
    for el in root.iter():
        cls = el.get('class')
        if cls in ('Note', 'Rest'):
            # A glyph drawn at its own size is a plain path; a scaled one
            # (a grace or cue note) is placed by its transform.
            m = re.search(r'matrix\(([^)]*)\)', el.get('transform') or '')
            if m:
                values = [float(v) for v in m.group(1).replace(',', ' ').split()]
                symbols.append((values[4] / scale, values[5] / scale))
            elif m := re.match(r'\s*M\s*([-\d.]+)[ ,]([-\d.]+)', el.get('d') or ''):
                symbols.append((float(m.group(1)) / scale, float(m.group(2)) / scale))
            continue
        if cls not in ('StaffLines', 'BarLine') or el.get('points') is None:
            continue
        pts = [tuple(float(v) / scale for v in p.split(',')) for p in el.get('points').split()]
        (xa, ya), (xb, yb) = pts[0], pts[-1]
        if cls == 'StaffLines':
            lines.append((ya, min(xa, xb), max(xa, xb)))
        else:
            bars.append((xa, min(ya, yb), max(ya, yb)))
    vb = [float(v) for v in root.get('viewBox').split()]
    width, height = int(round(vb[2] / scale)), int(round(vb[3] / scale))
    lines.sort()
    staves: list[Staff] = []
    group: list[tuple] = []
    for ln in lines:
        if group and (abs(ln[1] - group[0][1]) > 2 or abs(ln[2] - group[0][2]) > 2 or len(group) == 5):
            if len(group) == 5:
                staves.append(Staff([g[0] for g in group], group[0][1], group[0][2]))
            group = []
        group.append(ln)
    if len(group) == 5:
        staves.append(Staff([g[0] for g in group], group[0][1], group[0][2]))
    staves.sort(key=lambda s: s.top)
    per_staff = []
    for s in staves:
        xs = sorted({round(x, 1) for x, y0, y1 in bars if y0 <= s.center <= y1})
        per_staff.append(_merge(xs, s.space))
    systems: list[System] = []
    for i, s in enumerate(staves):
        same = (systems and per_staff[i] and per_staff[systems[-1].staves[-1]]
                and len(per_staff[i]) == len(per_staff[systems[-1].staves[-1]])
                and all(abs(a - b) < 2 for a, b in zip(per_staff[i], per_staff[systems[-1].staves[-1]]))
                and abs(s.x0 - staves[systems[-1].staves[-1]].x0) < 2)
        if same:
            systems[-1].staves.append(i)
        else:
            systems.append(System([i]))
    for system in systems:
        start = min(staves[i].x0 for i in system.staves)
        space = staves[system.staves[0]].space
        top = staves[system.staves[0]].top - 4 * space
        bottom = staves[system.staves[-1]].bottom + 4 * space
        # A barline before the system's first note or rest, such as a
        # repeat that opens the system after its clef and key, ends no bar.
        first = min((x for x, y in symbols if top <= y <= bottom), default=start)
        system.barlines = [x for x in per_staff[system.staves[0]] if x > start + 1.5 * space and x > first]
    return Layout(width, height, staves, systems)


def _merge(xs: list[float], space: float) -> list[float]:
    """A double barline or repeat sign, drawn as two strokes, taken once,
    at its last stroke."""
    out: list[float] = []
    for x in xs:
        if out and x - out[-1] <= 1.0 * space:
            out[-1] = x
        else:
            out.append(x)
    return out


@dataclass
class Counted:
    """A page's layout as a person counts it: for each system, its staves
    and its bars."""
    systems: list[tuple[int, int]]  # (staves, measures)


@dataclass
class LayoutScore:
    staves_found: int = 0
    staves_right: int = 0
    staves_true: int = 0
    line_error: float = 0.0  # mean distance of matched staff lines, in staff spaces
    systems_true: int = 0
    systems_right: int = 0  # the same staves exactly
    barlines_true: int = 0
    barlines_found: int = 0
    barlines_right: int = 0
    measures_right: int = 0  # systems with exactly the right number of bars
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        def f1(tp, found, true):
            p = tp / found if found else 1.0
            r = tp / true if true else 1.0
            return round(2 * p * r / (p + r), 4) if p + r else 0.0
        return {
            'staves_f1': f1(self.staves_right, self.staves_found, self.staves_true),
            'systems_exact': round(self.systems_right / self.systems_true, 4) if self.systems_true else 1.0,
            'barlines_f1': f1(self.barlines_right, self.barlines_found, self.barlines_true),
            'bars_per_system_exact': round(self.measures_right / self.systems_true, 4) if self.systems_true else 1.0,
            'line_error_spaces': round(self.line_error, 3),
            'staves': f'{self.staves_found} found, {self.staves_true} true',
            'notes': self.notes,
        }


def compare(truth: Layout, found: Layout) -> LayoutScore:
    """How a found layout matches an exact one, by position."""
    out = LayoutScore(staves_found=len(found.staves), staves_true=len(truth.staves),
                      systems_true=len(truth.systems))
    space = truth.space or 1.0
    match: dict[int, int] = {}
    errors = []
    for i, t in enumerate(truth.staves):
        best = min(range(len(found.staves)), key=lambda j: abs(found.staves[j].center - t.center), default=None)
        if best is not None and abs(found.staves[best].center - t.center) < 1.0 * space and best not in match.values():
            match[i] = best
            errors += [abs(a - b) / space for a, b in zip(t.lines, found.staves[best].lines)]
    out.staves_right = len(match)
    out.line_error = sum(errors) / len(errors) if errors else 0.0
    found_system = {}
    for k, s in enumerate(found.systems):
        for j in s.staves:
            found_system[j] = k
    for t in truth.systems:
        mapped = [match.get(i) for i in t.staves]
        ks = {found_system.get(j) for j in mapped}
        ok = None not in mapped and len(ks) == 1 and None not in ks and \
            len(found.systems[next(iter(ks))].staves) == len(t.staves)
        out.barlines_true += len(t.barlines)
        if ok:
            out.systems_right += 1
            f = found.systems[next(iter(ks))]
            out.barlines_found += len(f.barlines)
            used = set()
            for x in t.barlines:
                hit = next((j for j, y in enumerate(f.barlines) if j not in used and abs(x - y) < 1.0 * space), None)
                if hit is not None:
                    used.add(hit)
                    out.barlines_right += 1
            if len(f.barlines) == len(t.barlines) and len(used) == len(t.barlines):
                out.measures_right += 1
        else:
            out.notes.append(f'system at staff {t.staves[0]}: staves {mapped} found in systems {sorted(k for k in ks if k is not None)}')
    # Barlines of systems that matched nothing were found but are not right.
    matched = {found_system.get(match.get(t.staves[0])) for t in truth.systems}
    out.barlines_found += sum(len(s.barlines) for k, s in enumerate(found.systems) if k not in matched)
    return out


def compare_counted(truth: Counted, found: Layout) -> LayoutScore:
    """How a found layout matches what a person counted on the page."""
    out = LayoutScore(staves_true=sum(s for s, _ in truth.systems), staves_found=len(found.staves),
                      systems_true=len(truth.systems))
    got = [(len(s.staves), s.measures) for s in found.systems]
    out.staves_right = min(out.staves_true, out.staves_found)
    out.barlines_true = sum(m for _, m in truth.systems)
    out.barlines_found = sum(m for _, m in got)
    for k, (staves, measures) in enumerate(truth.systems):
        if k < len(got) and got[k][0] == staves:
            out.systems_right += 1
            out.barlines_right += min(measures, got[k][1])
            if got[k][1] == measures:
                out.measures_right += 1
            else:
                out.notes.append(f'system {k + 1}: {got[k][1]} bars found, {measures} counted')
        else:
            out.notes.append(f'system {k + 1}: {got[k] if k < len(got) else "nothing"} found, {(staves, measures)} counted')
    if len(got) != len(truth.systems):
        out.notes.append(f'{len(got)} systems found, {len(truth.systems)} counted')
    return out
