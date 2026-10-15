"""Our own reader: the layout stage finds the staves, the staff reader
reads each one, and the staves are joined into parts.

The model is the checkpoint STAFFREADER_MODEL names, or the best one of
the newest training run. Its name and step are part of NAME, so the
suite caches each model's readings apart from every other's.

A page's staves are read top to bottom. The layout groups each system's
staves into parts (a braced grand staff is one part on two staves), and
every system of a page is taken to hold the same parts in the same
order, so the k-th group of each system is the k-th part; each staff's
bars follow on, system after system."""

from __future__ import annotations

import collections
import os
import pathlib
from fractions import Fraction

import cv2
import torch

from .. import settings
from .. import layout, layout_suite, staffcrop, tokens
from .. import lyrics
from .. import model as M
from ..render import CACHE


# The model that reads unless STAFFREADER_MODEL names another: the v2 run
# of 2026-10-14, which reads every mark of the inventory as a second task.
# Against m4b-mix on every test set it was better beyond noise in 39 of
# 70 slices and worse in one (treble clef with an 8 below), and it halved
# the edits a staff needs (eval-v2-mix).
DEFAULT = CACHE / 'models/v2-mix/last.pt'


def _checkpoint() -> pathlib.Path | None:
    """The model to read with, or None on a machine that has none: the
    module's own logic still serves, and reading says what is missing."""
    named = os.environ.get('STAFFREADER_MODEL')
    if named:
        return pathlib.Path(named)
    return DEFAULT if DEFAULT.exists() else None


CHECKPOINT = _checkpoint()
_STATE = torch.load(CHECKPOINT, map_location='cpu', weights_only=True) if CHECKPOINT else {}
STEP = _STATE.get('step', 0)
# A checkpoint trained on both tasks reads each staff twice, the notes
# alone and then the marks, and the marks are put on the first reading's
# notes: weaving marks into a reading cost note accuracy (E8, E8c).
TASKS = bool(_STATE.get('tasks'))
del _STATE
# Greedy, no bar past twice its time signature; braced staves one part; one
# key per system; as many bars as the layout counts; an overfull bar read
# again as triplets (Reader.fits) within TRIPLET_MARGIN.
TRIPLET_MARGIN = float(settings.get('triplet_margin'))
DECODER = 'bars2-parts-key2-count' + ('-tasks' if TASKS else '') + (f'-triplets{TRIPLET_MARGIN:g}' if TRIPLET_MARGIN >= 0 else '')
# Nats a triplet reading of an overfull bar may fall short of the reading it
# replaces. The model cannot see a "3" that is not printed, so it always
# finds the triplets less likely; E11 (experiments/e11_refit.py) chose 12 on
# held-out staves: quiet triplets 46.8% to 58.6% of bars exact, real staves
# level. The triplet_margin setting changes it.
NAME = f'staffreader-{CHECKPOINT.parent.name if CHECKPOINT else "none"}-{STEP}-{DECODER}'
_NET = None


def net() -> M.Reader:
    global _NET
    if _NET is None:
        if CHECKPOINT is None:
            raise RuntimeError(f'no model at {DEFAULT}: set STAFFREADER_MODEL')
        _NET = M.load(CHECKPOINT)  # on the device setting's device (model.DEVICE)
    return _NET


def agree_on_key(crops: list, seqs: list[list[str]], limits: list | None = None, before: list | None = None,
                 floor: float = 0.05, known: float = 0.005, counts: list | None = None,
                 task: str | None = None) -> list[list[str]]:
    """One key across a system's staves. A staff whose key differs from
    the one most staves read is read again with that key, but only where
    the model itself gave that key a fair chance: [floor], or the lower
    [known] where the same part had that key in the system before
    ([before]). A transposing instrument, whose own key the model reads
    with confidence and keeps from system to system, keeps it."""
    keys = [next((t for t in seq if t.startswith('key:')), None) for seq in seqs]
    counts = collections.Counter(k for k in keys if k)
    if not counts:
        return seqs
    key, n = counts.most_common(1)[0]
    if n < 2 or n <= len(seqs) / 2:
        return seqs
    odd = [i for i, k in enumerate(keys) if k and k != key and seqs[i] and seqs[i][0].startswith('clef:')]
    if not odd:
        return seqs
    reader = net()
    device = next(reader.parameters()).device
    views = [M.window(crops[i]) for i in odd]
    x, widths = M.batch_images(views, device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        lead = [f'task:{task}'] if task else []
        chance = reader.chances(x, widths, [lead + [seqs[i][0]] for i in odd])[:, M.INDEX[key]]
    redo = [(i, k) for k, i in enumerate(odd)
            if chance[k].item() >= (known if before and before[i] == key else floor)]
    if not redo:
        return seqs
    x, widths = M.batch_images([views[k] for _, k in redo], device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        again = reader.read(x, widths, bar_limits=[limits[i] for i, _ in redo] if limits else None,
                            prefixes=[lead + [seqs[i][0], key] for i, _ in redo],
                            bar_counts=[counts[i] for i, _ in redo] if counts else None)
    out = list(seqs)
    for (i, _), seq in zip(redo, again):
        out[i] = [t for t in seq if not t.startswith('task:')]
    return out


def triplets(crops: list, seqs: list[list[str]], limits: list | None = None, task: str | None = None,
             margin: float | None = None) -> list[list[str]]:
    """Each reading with its overfull bars read again as triplets, where
    the model finds that within [margin] of what it read (Reader.fits)."""
    reader = net()
    device = next(reader.parameters()).device
    x, widths = M.batch_images([M.window(c) for c in crops], device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        fits = reader.fits(x, widths, seqs, limits=limits, lead=[[f'task:{task}']] * len(seqs) if task else None,
                           most=1.0, skip=M.meter_changes(seqs, limits))
    return [M.apply_fits(seq, f, TRIPLET_MARGIN if margin is None else margin) for seq, f in zip(seqs, fits)]


def read_staves(crops: list, batch: int = 12, limits: list | None = None,
                counts: list | None = None, task: str | None = None) -> list[list[str]]:
    """Each stored-size staff crop's tokens; [limits] are the bar lengths
    in force where each staff begins, where known, and [counts] the bars
    the layout found on each."""
    reader = net()
    device = next(reader.parameters()).device
    views = [M.window(c) for c in crops]
    order = sorted(range(len(views)), key=lambda i: views[i].shape[1])  # like widths together
    out: list[list[str] | None] = [None] * len(views)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        x, widths = M.batch_images([views[i] for i in idx], device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            got = reader.read(x, widths, bar_limits=[limits[i] for i in idx] if limits else None,
                              bar_counts=[counts[i] for i in idx] if counts else None,
                              prefixes=[[f'task:{task}']] * len(idx) if task else None)
            got = [[t for t in g if not t.startswith('task:')] for g in got]
        for i, seq in zip(idx, got):
            out[i] = seq
    return out


def read(image: pathlib.Path, page: int | None, work: pathlib.Path) -> pathlib.Path:
    work.mkdir(parents=True, exist_ok=True)
    gray = layout_suite.gray_page(image, page)
    lay = layout.read(gray)
    level = layout.level(gray, lay.angle, lay.bend)
    if not lay.systems:
        raise RuntimeError('no staves found on the page')
    # System by system, so each part's time signature, printed only where
    # it starts or changes, bounds its bars in every later system.
    parts: dict[int, dict[int, list[tokens.Bar]]] = {}  # part -> staff -> its bars
    # With a text reader, each staff's part comes from its label, so a
    # system that leaves out resting staves keeps every staff on its part.
    plan, named = _plan(level, lay)
    times: dict[tuple[int, int], object] = {}
    keys: dict[tuple[int, int], str] = {}  # each staff's key in the system before
    where, seqs = [], []
    for s_i, system in enumerate(lay.systems):
        groups = system.parts or [[st] for st in system.staves]
        ids = plan[s_i] if plan else list(range(len(groups)))
        ids = [k if k is not None else 1000 + g for g, k in enumerate(ids)]  # an unplaced staff a part of its own
        places = [(ids[g], n) for g, group in enumerate(groups) for n in range(len(group))]
        staves = [st for group in groups for st in group]
        crops = [staffcrop.crop(level, lay.staves[st].lines, lay.staves[st].x0, lay.staves[st].x1) for st in staves]
        limits = [times.get(place) for place in places]
        counts = [system.measures] * len(crops)
        task = 'notes' if TASKS else None
        got = agree_on_key(crops, read_staves(crops, limits=limits, counts=counts, task=task), limits,
                           [keys.get(place) for place in places], counts=counts, task=task)
        if TRIPLET_MARGIN >= 0:
            # A page that continues a work prints no time signature: its bars
            # are taken to fill the length most of them fill (tokens._meter).
            meter = tokens._meter([tokens.parse(seq) for seq in got])
            guess = None
            if meter:
                beats, unit, _ = tokens._time_value(meter)
                guess = Fraction(4 * beats, unit)
            # A staff whose bar length is not the one most of the system's bars
            # fill has a time signature that is wrong or not yet known: left alone.
            got = triplets(crops, got, [guess if lim is None or lim == guess else None for lim in limits], task)
        marked = read_staves(crops, limits=limits, counts=counts, task='marks') if TASKS else got
        for place, st, seq, full in zip(places, staves, got, marked):
            bars = tokens.parse(seq)
            if TASKS:
                bars = tokens.merge_marks(bars, tokens.parse(full))
            if 'ly' in full and lyrics.text_reader() is not None:
                staff = lay.staves[st]
                below = lay.staves[st + 1].top if st + 1 < len(lay.staves) else level.shape[0]
                verses = lyrics.read_rows(level, staff, lyrics.rows(level, staff, below))
                lyrics.attach(bars, verses)
            for b in bars:
                if b.time:
                    beats, unit, _ = tokens._time_value(b.time)
                    times[place] = Fraction(4 * beats, unit)
            keys[place] = next((t for t in seq if t.startswith('key:')), keys.get(place))
            parts.setdefault(place[0], {}).setdefault(place[1], []).extend(bars)
            where.append((s_i, *place))
            seqs.append(seq)
        if plan:  # a part this system leaves out rests through it
            present = {k for k, _ in places}
            for k in {k for k in parts} - present:
                for n in parts[k]:
                    parts[k][n] += [tokens.Bar(notes=[{'rest': True, 'measure': True, 'space': False, 'pitch': None,
                                                       'type': 'whole', 'dots': 0, 'tuplet': None, 'tie': False,
                                                       'voice': 0}]) for _ in range(system.measures)]
    (work / 'tokens.txt').write_text('\n'.join(f'{w}: {" ".join(s)}' for w, s in zip(where, seqs)))
    out = work / 'reading.musicxml'
    order = sorted(parts)
    for k in order:
        if named and k < len(named) and named[k].name == 'tenor':
            tenor_clef([parts[k][n] for n in sorted(parts[k])])
    out.write_bytes(tokens.to_musicxml([[parts[k][n] for n in sorted(parts[k])] for k in order],
                                       names=[named[k].key if named and k < len(named) else f'Part {i + 1}'
                                              for i, k in enumerate(order)],
                                       transpose=[named[k].transpose if named and k < len(named) else 0
                                                  for k in order]))
    return out


def tenor_clef(staves: list[list[tokens.Bar]]) -> None:
    """A tenor part printed in a plain treble clef, as older editions print
    one, sounds an octave below what it shows. It is written as the
    transcriptions write it: in the treble clef with an 8 below, an octave
    lower. A staff whose treble clef already shows its 8 is left as read.
    It applies where the text reader names the part, which on a page that
    continues a song needs the names carried from where they are printed."""
    for bars in staves:
        lower = False
        for bar in bars:
            if bar.clef:
                lower = bar.clef == 'G2'
                if lower:
                    bar.clef = 'G2-8'
            if lower:
                for n in bar.notes:
                    if n.get('pitch'):
                        n['pitch'] = n['pitch'][0] + str(int(n['pitch'][1:]) - 1)


def _plan(level, lay) -> tuple[list[list[int | None]] | None, list | None]:
    """Each system's part for each of its staff groups, from the labels
    the text reader reads at the system's left; or (None, None) without
    a text reader, or where no label is read at all. A label is read line
    by line ('Cor.' over '(C)'), and one printed between two staff groups
    names both, numbered in order ('Vl.' for the first and second
    violins). A braced pair whose label names no keyboard or harp is two
    players' staves, so each becomes a part of its own."""
    from .. import instruments
    net = lyrics.text_reader()
    if net is None:
        return None, None
    import torch
    from .. import text
    found, crops, owners = [], [], []
    for s_i, system in enumerate(lay.systems):
        groups = system.parts or [[st] for st in system.staves]
        labels = _margin_labels(level, lay, system, groups)
        found.append(labels)
        for k, (_, lines) in enumerate(labels):
            for line in lines:
                crops.append(text.view(line))
                owners.append((s_i, k))
    read = collections.defaultdict(list)
    got: dict[int, str] = {}
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1])  # like widths together: little padding
    for k in range(0, len(order), 32):
        idx = order[k:k + 32]
        x, w = text.batch([crops[i] for i in idx], next(net.parameters()).device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            got.update(zip(idx, net.read(x, w)))
    for i, owner in enumerate(owners):  # each label's lines in order, top to bottom
        read[owner].append(got[i])
    systems = []
    for s_i, system in enumerate(lay.systems):
        groups = system.parts or [[st] for st in system.staves]
        names: list[str | None] = [None] * len(groups)
        own: dict[int, str] = {}  # what stands beside one staff group alone, such as a horn's key
        for k, (covered, _) in enumerate(found[s_i]):
            label = ' '.join(t for t in read[(s_i, k)] if t).strip() or None
            if label and len(covered) == 1 and instruments.instrument(label) is None:
                own[covered[0]] = label
                continue
            for n, g in enumerate(covered):
                names[g] = f'{label} {n + 1}' if label and len(covered) > 1 else label
        for g, extra in own.items():  # 'Hr.' between the horns, 'C' and 'F' beside each
            names[g] = f'{names[g]} {extra}' if names[g] else extra
        # A braced pair that is no keyboard or harp: two players, two parts.
        split_groups, split_names = [], []
        for g, name in zip(groups, names):
            inst = instruments.instrument(name)
            if len(g) > 1 and inst is not None and inst.name not in ('piano', 'organ', 'harp'):
                split_groups += [[st] for st in g]
                split_names += [f'{name} {n + 1}' for n in range(len(g))]
            else:
                split_groups.append(g)
                split_names.append(name)
        if len(split_groups) != len(groups):
            system.parts = split_groups
        systems.append(split_names)
    if not any(instruments.instrument(l) for names in systems for l in names):
        return None, None
    parts, plan = instruments.assign(systems)
    return plan, parts


def _margin_labels(level, lay, system, groups) -> list[tuple[list[int], list]]:
    """The labels printed left of a system: for each, the staff groups it
    names (one, or the two it stands between) and a crop of each of its
    lines, cut close as the text reader's lines are. Some editions set a
    label hard against the bracket; brackets and braces are told from
    letters by height and taken out."""
    import cv2
    import numpy as np
    first, last = lay.staves[system.staves[0]], lay.staves[system.staves[-1]]
    space = first.space
    left = min(lay.staves[st].x0 for st in system.staves)
    top, bottom = max(0, int(first.top - space)), int(last.bottom + space)
    x0, x1 = max(0, int(left - 16 * space)), int(left - 0.3 * space)
    if x1 - x0 < 2 * space:
        return []
    strip = level[top:bottom, x0:x1].copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats((strip < 128).astype(np.uint8), connectivity=8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_HEIGHT] > 2.5 * space:
            strip[lab == k] = 255
    rows = np.flatnonzero((strip < 128).any(axis=1))
    if not len(rows):
        return []
    # Lines of text: runs of inked rows, a gap of a quarter space joining
    # what a line's dots and accents leave apart.
    lines, start = [], rows[0]
    for a, b in zip(rows, rows[1:]):
        if b - a > 0.25 * space:
            lines.append((start, a))
            start = b
    lines.append((start, rows[-1]))
    lines = [(a, b) for a, b in lines if b - a >= 0.4 * space]
    # Two lines whose parentheses or descenders touch read as one tall line:
    # split it at its thinnest row in the middle.
    ink = (strip < 128).sum(axis=1)
    tall = 1.6 * float(np.median([b - a for a, b in lines])) if lines else 0
    split = []
    for a, b in lines:
        if b - a > tall:
            mid = a + (b - a) // 4 + int(np.argmin(ink[a + (b - a) // 4:b - (b - a) // 4 + 1]))
            split += [(a, mid), (mid + 1, b)]
        else:
            split.append((a, b))
    lines = [(a, b) for a, b in split if b - a >= 0.4 * space]
    # Lines close together are one label.
    labels, current = [], []
    for line in lines:
        if current and line[0] - current[-1][1] > 0.8 * space:
            labels.append(current)
            current = []
        current.append(line)
    if current:
        labels.append(current)
    spans = [(lay.staves[g[0]].top, lay.staves[g[-1]].bottom) for g in groups]
    out = []
    for label in labels:
        y = top + (label[0][0] + label[-1][1]) / 2
        inside = [k for k, (t, b) in enumerate(spans) if t - 0.5 * space <= y <= b + 0.5 * space]
        between = [k for k in range(len(spans) - 1) if spans[k][1] < y < spans[k + 1][0]]
        nearest = min(range(len(spans)), key=lambda k: abs((spans[k][0] + spans[k][1]) / 2 - y))
        covered = inside[:1] or ([between[0], between[0] + 1] if between else [nearest])
        pad = int(round(0.4 * space))
        crops = []
        for a, b in label:
            cols = np.flatnonzero((strip[a:b + 1] < 128).any(axis=0))
            crops.append(strip[max(0, a - pad):b + pad + 1, max(0, cols[0] - pad):cols[-1] + pad + 1])
        out.append((covered, crops))
    return out
