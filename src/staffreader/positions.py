"""Where each note of a reading is printed on its page.

The staff reader writes what a staff holds, bar by bar, but not where on
the page each note sits. Anything found beside the music by other means
(directions in words, hairpins, fingering) is placed by the note it stands
at, so the reading's notes are matched to the page's noteheads:

1. heads: the noteheads of each staff, found by shape once the staff
   lines are taken out (a head is about a staff space tall and a little
   wider; a hollow head is filled first; stems and beams are too thin to
   stay).
2. place: bar by bar, the reading's onsets in time order are aligned with
   the heads' columns from left to right, each pair scored by how well the
   column's heights on the staff agree with the onset's pitches. Columns
   left over are clefs, accidentals and grace notes; an onset left over is
   one whose head was not found.

Each placed note gets 'x' and 'y', in the page's pixels."""

from __future__ import annotations

import re
from fractions import Fraction

import cv2
import numpy as np

from . import layout, tokens

# The note on each clef's bottom line, as (letter, octave).
BOTTOM = {'G2': ('E', 4), 'F4': ('G', 2), 'C3': ('F', 3), 'C4': ('D', 3), 'C1': ('C', 4), 'C2': ('A', 3),
          'F3': ('B', 2), 'G2-8': ('E', 3), 'G2+8': ('E', 5), 'F4-8': ('G', 1), 'percussion': ('E', 4)}
LETTERS = 'CDEFGAB'


def step(pitch: str, clef: str | None) -> int | None:
    """A pitch's height on the staff in steps above the bottom line (a line
    or a space each), under [clef]."""
    m = re.match(r'([A-G])[#bxn]*(-?\d)', pitch or '')
    letter, octave = BOTTOM.get(clef or 'G2', BOTTOM['G2'])
    if not m:
        return None
    return 7 * int(m.group(2)) + LETTERS.index(m.group(1)) - (7 * octave + LETTERS.index(letter))


def heads(level: np.ndarray, lay) -> dict[int, list[tuple[float, float]]]:
    """Each staff's noteheads, by the staff's index in the layout, as
    (x, y) centres in pixels."""
    ink = layout.binarize(level)
    line, _ = layout.run_lengths(ink)
    clean = layout.remove_staff_lines(ink, lay.staves, max(1, line))
    out: dict[int, list[tuple[float, float]]] = {}
    for i, st in enumerate(lay.staves):
        sp = st.space
        top = int(max(0, st.top - 5 * sp))
        band = clean[top:int(st.bottom + 5 * sp), int(st.x0):int(st.x1)].copy()
        # A hollow head's inside is a small enclosed patch of paper: filled.
        paper = (band == 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=4)
        small = [k for k in range(1, n) if stats[k, 4] < 0.8 * sp * sp and stats[k, 0] > 0 and stats[k, 1] > 0]
        if small:
            band[np.isin(lab, small)] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, round(0.95 * sp)), max(3, round(0.7 * sp))))
        opened = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
        n, _, stats, cent = cv2.connectedComponentsWithStats(opened, connectivity=8)
        found = []
        for k in range(1, n):
            w, h = stats[k, 2], stats[k, 3]
            if not (0.8 * sp <= w <= 2.4 * sp and h >= 0.6 * sp):
                continue
            rows = max(1, int(round(h / sp)))  # a chord's heads stacked a step or more apart
            for r in range(rows):
                found.append((st.x0 + cent[k][0], top + stats[k, 1] + (r + 0.5) * h / rows))
        out[i] = sorted(found)
    return out


def place(bars: list, staff, found: list[tuple[float, float]], ends: list[float]) -> int:
    """Give the notes of [bars] (one staff's reading of one system) the
    places of their heads among [found], the bars ending at the barlines
    [ends] (left to right, in pixels). Returns how many notes were placed."""
    sp = staff.space
    starts = [staff.x0] + list(ends[:-1])
    clef, placed = None, 0
    for b, bar in enumerate(bars):
        if bar.clef:
            clef = bar.clef
        if b >= len(ends):
            break
        onsets = tokens._onsets(bar.notes)
        events: dict[Fraction, list[dict]] = {}
        for n, at in zip(bar.notes, onsets):
            if n.get('rest') or n.get('grace') or n.get('space') or not n.get('pitch'):
                continue
            events.setdefault(at, []).append(n)
        order = sorted(events)
        inside = [h for h in found if starts[b] <= h[0] < ends[b]]
        columns: list[list[tuple[float, float]]] = []
        for h in inside:
            if columns and h[0] - columns[-1][0][0] < 0.7 * sp:
                columns[-1].append(h)
            else:
                columns.append([h])
        heights = [{round((staff.bottom - y) / (sp / 2)) for _, y in col} for col in columns]
        want = [{step(n['pitch'], n.get('clef') or clef) for n in events[at]} - {None} for at in order]
        # Align onsets with columns, both in order; a column may be passed
        # over cheaply (a clef, an accidental), an onset at a higher cost.
        E, C = len(order), len(columns)
        cost = np.full((E + 1, C + 1), np.inf)
        cost[0, :] = np.arange(C + 1) * 0.2
        back = {}
        for i in range(E + 1):
            for j in range(C + 1):
                if i == 0 and j == 0:
                    continue
                best = cost[i, j]
                if j > 0 and cost[i, j - 1] + 0.2 < best:
                    best, back[(i, j)] = cost[i, j - 1] + 0.2, (i, j - 1, None)
                if i > 0 and cost[i - 1, j] + 1.0 < best:
                    best, back[(i, j)] = cost[i - 1, j] + 1.0, (i - 1, j, None)
                if i > 0 and j > 0:
                    near = sum(1 for s in want[i - 1] if any(abs(s - h) <= 1 for h in heights[j - 1]))
                    miss = 1 - near / max(1, len(want[i - 1]))
                    if miss < 1 and cost[i - 1, j - 1] + miss < best:
                        best, back[(i, j)] = cost[i - 1, j - 1] + miss, (i - 1, j - 1, (i - 1, j - 1))
                cost[i, j] = best
        i, j = E, C
        while (i, j) in back:
            pi, pj, pair = back[(i, j)]
            if pair:
                e, c = pair
                col = columns[c]
                for n in events[order[e]]:
                    s = step(n['pitch'], n.get('clef') or clef)
                    x, y = min(col, key=lambda h: abs(round((staff.bottom - h[1]) / (sp / 2)) - (s or 0)))
                    n['x'], n['y'] = float(x), float(y)
                    placed += 1
            i, j = pi, pj
    return placed
