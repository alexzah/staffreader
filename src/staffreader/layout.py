"""The layout of a page: its staves, how they group into systems, and
where each system's barlines fall.

Everything after this works a staff or a system at a time, so the layout
must be right first. It is found the classical way, with no learning:

1. The page is evened out (its paper divided by a blurred copy of
   itself), thresholded, and turned straight by the angle at which its
   rows are most sharply black or white.
2. The thickness of a staff line and the space between lines are the most
   common vertical runs of black and of white.
3. Staff lines are what survives an opening with a long horizontal
   stroke; five of them evenly spaced make a staff.
4. Consecutive staves are one system when a line joins them at their left
   edge, as a system's opening barline or brace does.
5. A barline is a thin vertical stroke that fills a staff from its top
   line to its bottom line, with no notehead at either end, and that runs
   on into the next staff only where both are in one system. In a system
   of several staves a barline must be found in most of them.

A system's barlines are the bars' right edges: the line that opens a
system is not one, and a double barline or a repeat sign counts once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


def _scratch() -> pathlib.Path:
    """A folder for temporary files in the cache, never in /tmp, which is memory."""
    from .render import CACHE
    (CACHE / 'tmp').mkdir(parents=True, exist_ok=True)
    return CACHE / 'tmp'


@dataclass
class Staff:
    lines: list[float]  # the five lines' heights, top to bottom, in pixels
    x0: float
    x1: float

    @property
    def top(self) -> float:
        return self.lines[0]

    @property
    def bottom(self) -> float:
        return self.lines[-1]

    @property
    def space(self) -> float:
        return (self.bottom - self.top) / (len(self.lines) - 1)

    @property
    def center(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class System:
    staves: list[int]  # indices into Layout.staves
    parts: list[list[int]] = field(default_factory=list)  # the staves of each part, as braces join them
    barlines: list[float] = field(default_factory=list)  # bar ends, left to right, in pixels

    @property
    def measures(self) -> int:
        return len(self.barlines)


@dataclass
class Layout:
    width: int
    height: int
    staves: list[Staff]
    systems: list[System]
    angle: float = 0.0  # the rotation applied to straighten the page, in degrees
    bend: dict | None = None  # how find_bend flattened the page after turning it, if it did

    @property
    def space(self) -> float:
        return float(np.median([s.space for s in self.staves])) if self.staves else 0.0

    def to_dict(self) -> dict:
        return {'width': self.width, 'height': self.height, 'angle': round(self.angle, 3),
                'staves': [{'lines': [round(y, 1) for y in s.lines], 'x0': round(s.x0, 1), 'x1': round(s.x1, 1)}
                           for s in self.staves],
                'systems': [{'staves': s.staves, 'barlines': [round(x, 1) for x in s.barlines]} for s in self.systems]}


# ---------------------------------------------------------------- reading

def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink as 1 and paper as 0. The paper is evened out first, so a page
    darker at one side or yellowed thresholds as one."""
    g = gray.astype(np.float32)
    background = cv2.GaussianBlur(cv2.dilate(g, np.ones((15, 15), np.uint8)), (0, 0), 25)
    even = np.clip(g / np.maximum(background, 1) * 255, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(even, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Otsu's cut falls halfway between ink and paper, which loses lines
    # printed faint; anything clearly darker than the paper is ink, and
    # show-through from the other side stays lighter than this.
    cut = max(otsu, 190)
    return (even < cut).astype(np.uint8)


def straighten(ink: np.ndarray) -> tuple[np.ndarray, float]:
    """The page turned so its staff lines lie level, and the angle used.

    The page is cut into vertical strips. Each strip's rows of ink show
    the staff lines as a comb of peaks, and on a tilted page the comb
    sits a little lower or higher from one strip to the next. The shift
    of each strip against the middle one, found by correlating their
    rows, gives the slope."""
    h, w = ink.shape
    strips = 12
    # The strips span the music, not the paper: a last system that stops
    # short of the right margin still gives twelve strips to compare.
    cols = np.flatnonzero(ink.sum(axis=0) > 0.01 * h)
    left, right = (int(cols[0]), int(cols[-1]) + 1) if cols.size else (0, w)
    width = max(1, (right - left) // strips)
    profiles = [ink[:, left + i * width:left + (i + 1) * width].sum(axis=1).astype(np.float32) for i in range(strips)]
    centre = strips // 2
    reach = int(w * math.tan(math.radians(2.0)) / 2) + 2
    xs, shifts, weights = [], [], []
    ref = profiles[centre] - profiles[centre].mean()
    for i, p in enumerate(profiles):
        if i == centre or p.sum() < 0.002 * h * width:
            continue
        q = p - p.mean()
        lags = np.arange(-reach, reach + 1)
        scores = np.array([float(np.dot(ref[max(0, -d):h - max(0, d)], q[max(0, d):h - max(0, -d)])) for d in lags])
        best = int(scores.argmax())
        # A parabola through the peak gives the shift to a fraction of a row.
        frac = 0.0
        if 0 < best < len(lags) - 1:
            a, b, c = scores[best - 1], scores[best], scores[best + 1]
            if a - 2 * b + c != 0:
                frac = 0.5 * (a - c) / (a - 2 * b + c)
        xs.append((i - centre) * width)
        shifts.append(lags[best] + frac)
        weights.append(max(scores[best], 0.0))
    if len(xs) < 2:
        return ink, 0.0
    slope = float(np.polyfit(xs, shifts, 1, w=np.sqrt(np.array(weights) + 1e-9))[0])
    angle = math.degrees(math.atan(slope))
    if abs(angle) < 0.01:
        return ink, 0.0
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST), angle


def find_bend(ink: np.ndarray, period: int) -> dict | None:
    """How far each part of a level page sits above or below the middle,
    where the paper was not flat: a page that bends near its foot, say,
    tilts its last staves while the rest lie level.

    The page is cut into vertical strips and horizontal bands. In each
    band, each strip's rows of staff-line ink are matched against its
    neighbor's, nearer the middle, within half a line's period, so the
    comb of lines cannot slip a whole line; the shifts add up outward from
    the middle strip. Bands without staff lines take their shifts from the
    bands around them. A page that bends by less than a pixel and a half
    is left as it is (None)."""
    h, w = ink.shape
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (4 * period, 1)))
    strips = 8
    width = w // strips
    band = 12 * period
    centres = list(range(band // 2, h - band // 2 + 1, band // 2))
    if width < 6 * period or not centres:
        return None
    reach = max(1, period // 2)
    middle = strips // 2
    grid = np.full((len(centres), strips), np.nan)
    for b, yc in enumerate(centres):
        y0, y1 = yc - band // 2, yc + band // 2
        rows = [(lines[y0:y1, i * width:(i + 1) * width] > 0).sum(axis=1).astype(np.float32) for i in range(strips)]
        # A strip counts where it holds at least three lines' worth of ink.
        has = [r.sum() >= 3 * 0.5 * width for r in rows]
        if not has[middle]:
            continue
        grid[b, middle] = 0.0
        for side in (range(middle + 1, strips), range(middle - 1, -1, -1)):
            for i in side:
                j = i - 1 if i > middle else i + 1
                if not (has[i] and np.isfinite(grid[b, j])):
                    break
                ref, q = rows[j] - rows[j].mean(), rows[i] - rows[i].mean()
                n = len(ref)
                lags = np.arange(-reach, reach + 1)
                scores = np.array([float(np.dot(ref[max(0, -d):n - max(0, d)], q[max(0, d):n - max(0, -d)])) for d in lags])
                best = int(scores.argmax())
                frac = 0.0
                if 0 < best < len(lags) - 1:
                    x, y, z = scores[best - 1], scores[best], scores[best + 1]
                    if x - 2 * y + z != 0:
                        frac = 0.5 * (x - z) / (x - 2 * y + z)
                grid[b, i] = grid[b, j] + lags[best] + frac
    xs = np.array([(i + 0.5) * width for i in range(strips)])
    known_rows = []
    for b in range(len(centres)):
        ok = np.isfinite(grid[b])
        if ok.sum() >= strips // 2:
            grid[b] = np.interp(xs, xs[ok], grid[b, ok])
            known_rows.append(b)
        else:
            grid[b] = np.nan
    if not known_rows:
        return None
    ys = np.array(centres, float)
    for i in range(strips):
        grid[:, i] = np.interp(ys, ys[known_rows], grid[known_rows, i])
    if np.abs(grid).max() < 1.5:
        return None
    return {'xs': xs.tolist(), 'ys': ys.tolist(), 'shift': np.round(grid, 2).tolist()}


def unbend(image: np.ndarray, bend: dict | None, ink: bool = False) -> np.ndarray:
    """[image] with the bend find_bend measured taken out, so its staff
    lines run straight. An ink image is resampled by nearest pixel, a gray
    one smoothly onto white paper."""
    if not bend:
        return image
    h, w = image.shape
    xs, ys, grid = np.array(bend['xs']), np.array(bend['ys']), np.array(bend['shift'])
    across = np.stack([np.interp(np.arange(w), xs, row) for row in grid])  # each band's shift at every column
    field = np.empty((h, w), np.float32)
    for x in range(w):
        field[:, x] = np.interp(np.arange(h), ys, across[:, x])
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.arange(h, dtype=np.float32)[:, None] + field
    return cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST if ink else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0 if ink else 255)


def level(gray: np.ndarray, angle: float, bend: dict | None = None) -> np.ndarray:
    """A gray page turned by the angle straighten found, and flattened by
    the bend find_bend found, so it lines up with the layout read from it."""
    if angle != 0.0:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    return unbend(gray, bend)


def run_lengths(ink: np.ndarray) -> tuple[int, int]:
    """The most common vertical run of ink (a staff line's thickness) and
    of paper (the space between lines), sampled over many columns."""
    cols = ink[:, :: max(1, ink.shape[1] // 400)]
    black, white = np.zeros(200, np.int64), np.zeros(400, np.int64)
    for c in cols.T:
        change = np.flatnonzero(np.diff(np.concatenate(([0], c, [0]))))
        starts, ends = change[::2], change[1::2]
        lengths = ends - starts
        black += np.bincount(np.clip(lengths, 0, 199), minlength=200)
        gaps = starts[1:] - ends[:-1]
        white += np.bincount(np.clip(gaps, 0, 399), minlength=400)
    black[0] = white[0] = 0
    return int(black.argmax()), int(white.argmax())


def find_staves(ink: np.ndarray, line: int, space: int) -> list[Staff]:
    """Groups of five evenly spaced horizontal lines."""
    period = line + space
    # Noise and thresholding break a thin line into short pieces; small
    # gaps along a row are bridged before looking for long lines.
    bridged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, space // 2), 1)))
    long = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (6 * period, 1)))
    rows = long.sum(axis=1).astype(np.float32)
    width = ink.shape[1]
    candidates = np.flatnonzero(rows >= 6 * period)
    # Rows of ink side by side are one line, at their weighted center.
    lines = []
    if candidates.size:
        groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1)
        for g in groups:
            # A beam lying on a line thickens it; the line is the rows
            # nearly as long as the longest, since a beam is short.
            # A beam drawn very long, on a page with few notes to a line,
            # can pass that test and leave the line too thick; then only
            # the rows almost as long as the line itself are kept.
            for share in (0.7, 0.9):
                kept = g[rows[g] >= share * rows[g].max()]
                subs = [sub for sub in np.split(kept, np.flatnonzero(np.diff(kept) > 1) + 1)
                        if sub.size <= 3 * max(line, 1)]
                if subs:
                    break
            for sub in subs:
                weights = rows[sub]
                lines.append(float((sub * weights).sum() / weights.sum()))
    # Every run of five lines a staff space apart is a candidate, passing
    # over beams and ledger lines that fall between them. A ledger line can
    # make a false five with four real lines, so candidates are taken
    # longest first, the length a staff's lines share, and none may reuse
    # a line or overlap a staff already taken.
    candidates = []
    for i, y in enumerate(lines):
        five, picked = [y], [i]
        step = float(period)
        for k in range(1, 5):
            want = five[-1] + step
            j = min((j for j in range(i + 1, len(lines)) if j not in picked),
                    key=lambda j: abs(lines[j] - want), default=None)
            if j is None or abs(lines[j] - want) > 0.25 * period:
                break
            five.append(lines[j])
            picked.append(j)
            step = (five[-1] - five[0]) / k
        if len(five) < 5:
            continue
        # The staff runs where at least four of its lines have ink. Each of
        # its lines must run along most of that stretch, which a ledger
        # line taken for a staff line does not.
        per_line = [long[max(0, int(round(ly)) - line):int(round(ly)) + line + 1].max(axis=0) for ly in five]
        hits = np.sum(per_line, axis=0)
        cols = np.flatnonzero(hits >= 4)
        if cols.size < 6 * period:
            continue
        lengths = [int(p[cols[0]:cols[-1] + 1].sum()) for p in per_line]
        if min(lengths) < 0.5 * (cols[-1] - cols[0] + 1):
            continue
        candidates.append((sum(lengths), five, picked, int(cols[0]), int(cols[-1])))
    staves = []
    used: set[int] = set()
    for _, five, picked, a, b in sorted(candidates, key=lambda c: -c[0]):
        if used & set(picked) or any(five[0] <= s.bottom and s.top <= five[-1] for s in staves):
            continue
        x0, x1 = _extend(ink, five, a, b, line, period)
        staves.append(Staff(five, float(x0), float(x1)))
        used.update(picked)
    return sorted(staves, key=lambda s: s.top)


def _extend(ink: np.ndarray, five: list[float], x0: int, x1: int, line: int, period: int) -> tuple[int, int]:
    """A staff's ends, followed out from the long stretch found: on past
    clefs, key signatures and repeat signs that break its lines into short
    pieces, while most of its five lines still have ink, over gaps no wider
    than a staff space."""
    height = ink.shape[0]
    reach = line + 1

    def walk(x, step):
        # Each line is followed as it drifts up or down across the page,
        # as the lines of a scanned page do.
        ys = [int(round(y)) for y in five]
        gap, last = 0, x
        while 0 <= x + step < ink.shape[1] and gap <= period:
            x += step
            hits = 0
            for k, y in enumerate(ys):
                col = ink[max(0, y - reach):min(height, y + reach + 1), x]
                if col.any():
                    hits += 1
                    offsets = np.flatnonzero(col) + max(0, y - reach)
                    ys[k] = int(round(offsets.mean())) if abs(offsets.mean() - y) <= 1.5 else y
            if hits >= 3:
                gap, last = 0, x
            else:
                gap += 1
        return last

    return walk(x0, -1), walk(x1, 1)


def _joined(ink: np.ndarray, x: float, y0: float, y1: float, thickness: int, reach: int) -> bool:
    """Whether ink runs from height [y0] to [y1] near column [x]: the
    share of rows between them with ink within [reach] pixels of it."""
    a, b = max(0, int(x) - reach), min(ink.shape[1], int(x) + reach + 1)
    strip = ink[int(y0):int(y1) + 1, a:b]
    if strip.size == 0:
        return False
    return float((strip.max(axis=1) > 0).mean()) > 0.92


def find_systems(ink: np.ndarray, staves: list[Staff], line: int) -> list[System]:
    """Staves joined at their left edge by a line, a system's opening
    barline or bracket, are one system."""
    systems: list[System] = []
    current = [0] if staves else []
    for i in range(1, len(staves)):
        a, b = staves[i - 1], staves[i]
        x = min(a.x0, b.x0)
        space = (a.space + b.space) / 2
        # Where a staff's lines run on into a bracket or a brace, the line
        # that joins the system stands a few spaces right of where they seem
        # to start; nothing else crosses the gap there, before any clef ends.
        joined = any(_joined(ink, x + dx, a.bottom, b.top, line, max(2, line))
                     for dx in np.arange(-1.5 * space, 6 * space, max(1, line)))
        if joined:
            current.append(i)
        else:
            systems.append(System(current))
            current = [i]
    if current:
        systems.append(System(current))
    return systems


def braced(ink: np.ndarray, a: Staff, b: Staff) -> bool:
    """Whether a brace makes two neighboring staves one instrument's grand
    staff: a brace's stroke, which ends with the pair's outer lines, and no
    group bracket inside it (as a score braces the two horns or the two
    violins within their family's bracket)."""
    if not _brace_shape(ink, a, b):
        return False
    space = (a.space + b.space) / 2
    x0 = min(a.x0, b.x0)
    lo, hi = max(0, int(x0 - 3.5 * space)), int(x0 + 1.5 * space)
    band = ink[int(a.top):int(b.bottom) + 1, lo:hi] > 0
    rows = band.any(axis=1)
    left = np.where(rows, band.argmax(axis=1), 0)
    x = lo + int(np.median(left[rows]))
    # 1. A bracket grouping more staves runs on past the pair.
    reach = max(2, int(0.3 * space))
    for y0, y1 in ((a.top - 1.6 * space, a.top - 0.6 * space), (b.bottom + 0.6 * space, b.bottom + 1.6 * space)):
        strip = ink[max(0, int(y0)):max(0, int(y1)), max(0, x - reach):x + reach + 1]
        if strip.size and float((strip.max(axis=1) > 0).mean()) > 0.8:
            return False
    # 2. A family's bracket is a straight, thick stroke down the whole pair:
    # columns inked nearly all the way, side by side for a third of a
    # space. A brace drifts sideways as it curves, and the system's opening
    # line is thin.
    full = band.mean(axis=0) >= 0.85
    run = best = 0
    for f in full:
        run = run + 1 if f else 0
        best = max(best, run)
    if best >= 0.35 * space:
        return False
    return True


def _brace_shape(ink: np.ndarray, a: Staff, b: Staff) -> bool:
    """Whether a brace joins two neighboring staves, making them one
    instrument's grand staff. A brace curves, its middle reaching further
    left than its ends, or its stroke thins to a waist at the middle; a
    bracket, which groups instruments, is straight and evenly thick."""
    space = (a.space + b.space) / 2
    x0 = min(a.x0, b.x0)
    # The staff finder follows the lines out into whatever ink joins them,
    # so a brace may sit at or just right of a staff's left end.
    lo, hi = max(0, int(x0 - 3.5 * space)), int(x0 + 1.5 * space)
    band = ink[int(a.top):int(b.bottom) + 1, lo:hi]
    rows = band.max(axis=1) > 0
    if rows.mean() < 0.85:
        return False
    left = np.where(rows, band.argmax(axis=1), hi - lo).astype(float)
    h = len(left)
    middle = left[int(h * 0.4):int(h * 0.6) + 1].min()  # the cusp is narrow
    ends = np.median(np.concatenate([left[int(h * 0.15):int(h * 0.25) + 1], left[int(h * 0.75):int(h * 0.85) + 1]]))
    if ends - middle > 0.4 * space:
        return True
    # The width of the leftmost stroke, row by row.
    width = np.zeros(h)
    for r in range(h):
        if rows[r]:
            run = band[r, int(left[r]):] > 0
            width[r] = np.argmin(run) if not run.all() else len(run)
    waist = width[int(h * 0.4):int(h * 0.6) + 1].min()
    body = np.median(np.concatenate([width[int(h * 0.2):int(h * 0.35) + 1], width[int(h * 0.65):int(h * 0.8) + 1]]))
    return body > 0.25 * space and waist < 0.6 * body


ORCHESTRAL = 8  # staves in a system from which braces are not read as grand staves


def find_parts(ink: np.ndarray, staves: list[Staff], systems: list[System]):
    """Each system's staves grouped into parts: braced staves are one
    part, every other staff a part of its own. In an orchestral system an
    engraver also braces pairs of instruments inside a family's bracket
    (two horns, two violins), which a brace alone cannot tell from a
    grand staff; there every staff stays a part until brackets are read
    as well."""
    for system in systems:
        groups = [[system.staves[0]]] if system.staves else []
        orchestral = len(system.staves) >= ORCHESTRAL
        for i, j in zip(system.staves, system.staves[1:]):
            if not orchestral and braced(ink, staves[i], staves[j]) and len(groups[-1]) < 3:
                groups[-1].append(j)
            else:
                groups.append([j])
        system.parts = groups


def _vertical_segments(ink: np.ndarray, staff: Staff, line: int) -> list[tuple[int, int]]:
    """Runs of columns where ink fills the staff from top line to bottom
    line, as (first column, last column)."""
    y0, y1 = int(round(staff.top)), int(round(staff.bottom))
    x0, x1 = max(0, int(staff.x0) - 2), min(ink.shape[1], int(staff.x1) + 3)
    band = ink[max(0, y0 - line):y1 + line + 1, x0:x1]  # a staff at the page's top edge
    if band.size == 0:
        return []
    # A barline printed thin and broken, or wavering by a pixel, still
    # fills most of the staff within a column or so either side.
    band = cv2.dilate(band, np.ones((1, 3), np.uint8))
    filled = (band.sum(axis=0) >= 0.95 * (y1 - y0))
    cols = np.flatnonzero(filled)
    if cols.size == 0:
        return []
    runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    return [(int(r[0]) + x0, int(r[-1]) + x0) for r in runs]


def remove_staff_lines(ink: np.ndarray, staves: list[Staff], line: int) -> np.ndarray:
    """[ink] with the staff lines taken out: on each line, a column's ink
    is cleared where nothing continues above or below the line, so the
    noteheads and stems crossing it stay."""
    out = ink.copy()
    reach = max(1, line)
    for s in staves:
        a, b = int(s.x0), int(s.x1) + 1
        for y in s.lines:
            y0, y1 = int(round(y)) - reach, int(round(y)) + reach + 1
            alone = (ink[y0 - 1, a:b] == 0) & (ink[y1, a:b] == 0)
            band = out[y0:y1, a:b]
            band[:, alone] = 0
    return out


def _notehead_beside(clean: np.ndarray, staff: Staff, a: int, b: int) -> bool:
    """Whether a notehead touches the stroke [a, b] anywhere along the
    staff, which makes it a stem rather than a barline. [clean] has its
    staff lines removed, so only noteheads and other symbols are left."""
    space = staff.space
    width = max(2, int(round(0.6 * space)))
    tall = max(2, int(round(0.7 * space)))
    y0, y1 = int(staff.top - space), int(staff.bottom + space)
    # A notehead is joined to its stem; repeat dots stand apart from their
    # barline, so the columns right beside the stroke must carry ink too.
    for side, touch in ((clean[y0:y1, max(0, a - 1 - width):max(0, a - 1)], clean[y0:y1, max(0, a - 2):a]),
                        (clean[y0:y1, b + 2:b + 2 + width], clean[y0:y1, b + 1:b + 3])):
        if side.size == 0 or touch.size == 0:
            continue
        rows = np.convolve(side.mean(axis=1), np.ones(tall) / tall, mode='valid') if side.shape[0] >= tall else None
        if rows is None:
            continue
        at = np.convolve(touch.max(axis=1).astype(float), np.ones(tall) / tall, mode='valid')
        if np.any((rows > 0.5) & (at > 0.5)):
            return True
    return False


def find_barlines(ink: np.ndarray, staves: list[Staff], systems: list[System], line: int):
    clean = remove_staff_lines(ink, staves, line)
    for system in systems:
        found: list[list[tuple[float, bool]]] = []
        strokes = [_vertical_segments(ink, staves[si], line) for si in system.staves]

        def filled(k, x):
            """Whether staff [k] of the system is filled top to bottom at [x]."""
            reach = 0.5 * staves[system.staves[k]].space
            return any(a - reach <= x <= b + reach for a, b in strokes[k])

        for k, si in enumerate(system.staves):
            staff = staves[si]
            space = staff.space
            xs = []
            for a, b in strokes[k]:
                if b - a + 1 > 1.8 * space:  # too wide even for a closing double barline
                    continue
                x = (a + b) / 2
                above = _joined(ink, x, staff.top - 1.5 * space, staff.top - 2, line, 1)
                below = _joined(ink, x, staff.bottom + 2, staff.bottom + 1.5 * space, line, 1)
                into_next = k + 1 < len(system.staves) and _joined(
                    ink, x, staff.bottom, staves[system.staves[k + 1]].top, line, 1)
                into_prev = k > 0 and _joined(ink, x, staves[system.staves[k - 1]].bottom, staff.top, line, 1)
                if (above and not into_prev) or (below and not into_next):
                    continue  # a stem reaching past the staff
                # A stroke that runs on into every neighboring staff of its
                # system is a barline, whatever touches it (a slur, a
                # hairpin, a note pressed close); only one that stays in its
                # own staff is judged by the notehead beside it.
                # A chord's stem can also cross from one staff into the next
                # and end at a notehead there, so the neighbor must be filled
                # top to bottom at the same place.
                last = len(system.staves) - 1
                through = (last > 0 and (k == 0 or (into_prev and filled(k - 1, x)))
                           and (k == last or (into_next and filled(k + 1, x))))
                if not through and _notehead_beside(clean, staff, a, b):
                    continue
                xs.append((x, into_next or into_prev, _dots_after(clean, staff, b)))
            found.append(xs)
        system.barlines = _consensus(found, staves, system)


def _dots_after(clean: np.ndarray, staff: Staff, b: int) -> bool:
    """Whether a repeat's two dots stand just right of a stroke, in the
    second and third spaces of the staff. On a small staff the two run
    together, so they are told from a notehead on the middle line by their
    width: a dot is narrow, a notehead wider than a staff space."""
    space = staff.space
    half = max(1, int(round(0.25 * space)))
    x0, x1 = b + 1, b + 1 + int(round(1.6 * space))

    def dot_at(y):
        band = clean[int(y) - half:int(y) + half + 1, x0:x1]
        if band.size == 0:
            return False
        cols = np.flatnonzero(band.max(axis=0))
        if cols.size == 0 or cols[0] > 0.6 * space:
            return False
        run = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)[0]
        return run.size <= 0.75 * space

    return dot_at(staff.top + 1.5 * space) and dot_at(staff.top + 2.5 * space)


def _consensus(found: list[list[tuple[float, bool, bool]]], staves: list[Staff], system: System) -> list[float]:
    """The barlines most of a system's staves agree on, a double barline
    or repeat sign taken once, and the system's opening line left out.
    Where the system's barlines run on from staff to staff, as in a grand
    staff or a quartet, a stroke that stops at its own staff is a stem."""
    space = float(np.median([staves[i].space for i in system.staves]))
    every = sorted(x for xs in found for x, _, _ in xs)
    clusters: list[list[float]] = []
    for x in every:
        if clusters and x - clusters[-1][-1] <= 1.0 * space:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    def near(c, xs):
        return [(x, j, d) for x, j, d in xs if c[0] - 0.5 * space <= x <= c[-1] + 0.5 * space]

    need = max(1, math.ceil(len(found) / 2))
    kept = []
    for c in clusters:
        hits = [h for xs in found for h in near(c, xs)]
        support = sum(1 for xs in found if near(c, xs))
        if support >= need:
            # A repeat's or double bar's last stroke ends the bar.
            kept.append((max(c), any(j for _, j, _ in hits), any(d for _, _, d in hits)))
    if len(system.staves) > 1 and sum(j for _, j, _ in kept) >= 2:
        kept = [k for k in kept if k[1]]
    # The system's opening line stands at the staves' start, before the
    # clef, key and time, which take more than two staff spaces.
    start = float(np.median([staves[i].x0 for i in system.staves]))
    kept = [k for k in kept if k[0] > start + 2.5 * space]
    # A repeat that opens the system, just after its clef and key, starts
    # the first bar and ends none.
    if kept and kept[0][2] and kept[0][0] < start + 14 * space:
        kept = kept[1:]
    return [x for x, _, _ in kept]


def read(gray: np.ndarray) -> Layout:
    """The layout of a gray page image."""
    ink, angle = straighten(binarize(gray))
    line, space = run_lengths(ink)
    bend = find_bend(ink, max(1, line) + max(2, space))
    ink = unbend(ink, bend, ink=True)
    staves = find_staves(ink, max(1, line), max(2, space))
    systems = find_systems(ink, staves, max(1, line))
    find_parts(ink, staves, systems)
    find_barlines(ink, staves, systems, max(1, line))
    return Layout(gray.shape[1], gray.shape[0], staves, systems, angle, bend)


def read_file(path, page: int | None = None, dpi: int = 300) -> Layout:
    """The layout of a PNG, or of one page of a PDF rendered at [dpi]."""
    import pathlib
    import subprocess
    import tempfile
    path = pathlib.Path(path)
    if page is None:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        with tempfile.TemporaryDirectory(dir=_scratch()) as d:
            out = pathlib.Path(d) / 'page'
            subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-r', str(dpi), '-gray', '-singlefile',
                            '-png', str(path), str(out)], check=True)
            gray = cv2.imread(str(out) + '.png', cv2.IMREAD_GRAYSCALE)
    return read(gray)


def draw(gray: np.ndarray, layout: Layout, out) -> None:
    """The page with its layout drawn over it: staves in blue, systems
    bracketed in green, barlines in red."""
    img = cv2.cvtColor(level(gray, layout.angle, layout.bend), cv2.COLOR_GRAY2BGR)
    for s in layout.staves:
        for y in s.lines:
            cv2.line(img, (int(s.x0), int(y)), (int(s.x1), int(y)), (220, 120, 0), 2)
    for k, system in enumerate(layout.systems):
        top = layout.staves[system.staves[0]].top
        bottom = layout.staves[system.staves[-1]].bottom
        x0 = int(min(layout.staves[i].x0 for i in system.staves)) - 25
        cv2.line(img, (x0, int(top)), (x0, int(bottom)), (0, 160, 0), 6)
        cv2.putText(img, f'{system.measures}', (x0 - 60, int(top) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 160, 0), 3)
        for x in system.barlines:
            cv2.line(img, (int(x), int(top) - 20), (int(x), int(bottom) + 20), (0, 0, 230), 3)
    cv2.imwrite(str(out), img)
