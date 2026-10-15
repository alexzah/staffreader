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

from .. import layout, layout_suite, staffcrop, tokens
from .. import lyrics
from .. import model as M
from ..render import CACHE


# The model that reads unless STAFFREADER_MODEL names another: the M4b
# overnight run after its mixed phase, chosen 2026-10-14
# as the best all round, solo piano above all.
DEFAULT = CACHE / 'models/m4b-mix/last.pt'


def _checkpoint() -> pathlib.Path:
    named = os.environ.get('STAFFREADER_MODEL')
    if named:
        return pathlib.Path(named)
    if not DEFAULT.exists():
        raise RuntimeError(f'no model at {DEFAULT}: set STAFFREADER_MODEL')
    return DEFAULT


CHECKPOINT = _checkpoint()
_STATE = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
STEP = _STATE.get('step', 0)
# A checkpoint trained on both tasks reads each staff twice, the notes
# alone and then the marks, and the marks are put on the first reading's
# notes: weaving marks into a reading cost note accuracy (E8, E8c).
TASKS = bool(_STATE.get('tasks'))
del _STATE
DECODER = 'bars2-parts-key2-count' + ('-tasks' if TASKS else '')  # greedy, no bar past twice its time signature; braced staves one part; one key per system; as many bars as the layout counts
NAME = f'staffreader-{CHECKPOINT.parent.name}-{STEP}-{DECODER}'
_NET = None


def net() -> M.Reader:
    global _NET
    if _NET is None:
        _NET = M.load(CHECKPOINT, 'cuda' if torch.cuda.is_available() else 'cpu')
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
    out.write_bytes(tokens.to_musicxml([[parts[k][n] for n in sorted(parts[k])] for k in order],
                                       names=[named[k].key if named and k < len(named) else f'Part {i + 1}'
                                              for i, k in enumerate(order)],
                                       transpose=[named[k].transpose if named and k < len(named) else 0
                                                  for k in order]))
    return out


def _plan(level, lay) -> tuple[list[list[int | None]] | None, list | None]:
    """Each system's part for each of its staff groups, from the labels
    the text reader reads at the system's left; or (None, None) without
    a text reader, or where no label is read at all."""
    from .. import instruments
    net = lyrics.text_reader()
    if net is None:
        return None, None
    import torch
    from .. import text
    crops, owners = [], []
    for s_i, system in enumerate(lay.systems):
        groups = system.parts or [[st] for st in system.staves]
        left = min(lay.staves[st].x0 for st in system.staves)
        for g, group in enumerate(groups):
            a, b = lay.staves[group[0]], lay.staves[group[-1]]
            space = a.space
            # The label stands left of the bracket, level with its staves
            # or centred on a pair of them.
            y0, y1 = int(a.top - 1.0 * space), int(b.bottom + 1.0 * space)
            x0, x1 = max(0, int(left - 16 * space)), int(left - 1.5 * space)
            if x1 - x0 < 2 * space:
                continue
            crop = level[max(0, y0):y1, x0:x1]
            ink = (crop < 128).any(axis=0)
            if ink.sum() < space:
                continue
            crops.append(text.view(crop))
            owners.append((s_i, g))
    labels = {}
    for k in range(0, len(crops), 32):
        x, w = text.batch(crops[k:k + 32], next(net.parameters()).device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            for owner, label in zip(owners[k:k + 32], net.read(x, w)):
                labels[owner] = label
    if not any(instruments.instrument(l) for l in labels.values()):
        return None, None
    systems = [[labels.get((s_i, g)) for g in range(len(system.parts or system.staves))]
               for s_i, system in enumerate(lay.systems)]
    parts, plan = instruments.assign(systems)
    return plan, parts
