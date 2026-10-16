"""Lyrics on a page: the verses sung under a staff, read by the text
reader and put under the notes the staff reader marked as taking a
syllable ('ly').

A verse is a row of text below the staff: a horizontal band of ink about
a staff space tall that runs across much of the staff, which a dynamic
(a few marks) or a hairpin (a thin wedge) does not. Rows are verses in
order from the top. The text reader writes a verse as syllables with a
space between: a hyphen after one the next continues, an underscore
after one held over the notes that follow, and a hyphen before one that
carries on a word from the system before. Syllables go to the marked
notes in order."""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np

from .render import CACHE
from . import settings

# The text reader: e9b-real trained further on staff labels engraved as older
# editions print them and real labels read from paired orchestral pages
# (text-v3, 2026-10-16). Against e9b-real, real lyric lines read exactly
# went from 22% to 30% and characters wrong from 4.3% to 3.2%; rendered
# lines 91.7% to 92.8% exact.
TEXT_MODEL = settings.path('text_model') or CACHE / 'models' / 'text-v3' / 'last.pt'
_NET = None


def text_reader():
    """The text reader, or None where none has been trained."""
    global _NET
    if _NET is None and TEXT_MODEL.exists():
        from . import text
        _NET = text.load(TEXT_MODEL)
    return _NET


def rows(level: np.ndarray, staff, below: float) -> list[tuple[int, int]]:
    """The rows of text in the band under [staff], down to [below] (the
    next staff's top line, or where the page's music ends): (top, bottom)
    in pixels, top first."""
    space = staff.space
    y0, y1 = int(staff.bottom + 1.0 * space), int(min(below - 0.5 * space, staff.bottom + 16 * space))
    x0, x1 = int(staff.x0), int(staff.x1)
    if y1 - y0 < space:
        return []
    band = level[y0:y1, x0:x1] < 128
    # A slur, beam or stem reaching into the band would join a verse to the
    # staff below; no character or extender of a verse is that tall, nor
    # that long and thick at once, and none runs on into the staff below.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
    w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    keep = (h <= 2.5 * space) & ~((w > 6 * space) & (h > 0.5 * space))
    keep &= stats[:, cv2.CC_STAT_TOP] + h < band.shape[0]
    keep[0] = False
    band = keep[labels]
    rows_ink = band.sum(axis=1)
    on = rows_ink > 0.01 * band.shape[1]
    out, start = [], None
    for y, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = y
        elif not v and start is not None:
            top, bottom = start, y
            start = None
            if not 0.6 * space <= bottom - top <= 3.5 * space:
                continue
            cols = band[top:bottom].any(axis=0)
            # Text runs across the staff in words; a dynamic or a slur's end does not.
            if cols.mean() < 0.12 or _gaps(cols) < 3:
                continue
            out.append((y0 + top, y0 + bottom))
    out = _merge(out, space)
    # Verses under one staff share a size: a row much thinner than the
    # rest is the beams and stems hanging below the staff.
    if len(out) > 1:
        tall = sorted(b - a for a, b in out)[len(out) // 2]
        out = [(a, b) for a, b in out if b - a >= 0.6 * tall]
    return out


def _gaps(cols: np.ndarray) -> int:
    """How many separate runs of ink a row has across the staff."""
    return int(np.count_nonzero(np.diff(cols.astype(np.int8)) == 1)) + int(cols[0])


def _merge(found: list[tuple[int, int]], space: float) -> list[tuple[int, int]]:
    """Rows a hair apart (an accent above a capital, a descender) are one."""
    out: list[tuple[int, int]] = []
    for top, bottom in found:
        if out and top - out[-1][1] < 0.25 * space:
            out[-1] = (out[-1][0], bottom)
        else:
            out.append((top, bottom))
    return out


def read_rows(level: np.ndarray, staff, found: list[tuple[int, int]]) -> list[str]:
    """Each row's text, as the text reader writes it."""
    import torch
    from . import text
    net = text_reader()
    if net is None or not found:
        return []
    pad = int(0.4 * staff.space)
    views = [text.view(level[max(0, a - pad):b + pad, int(staff.x0):int(staff.x1) + 1]) for a, b in found]
    x, w = text.batch(views, next(net.parameters()).device)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        return net.read(x, w)


def syllables(line: str) -> list[tuple[str, str, bool]]:
    """A verse as (text, syllabic, extended) for each syllable, in the
    MusicXML sense of syllabic: single, begin, middle or end."""
    out = []
    going = False  # the word before goes on into this syllable
    for word in line.split():
        carried = word.startswith('-') and len(word) > 1
        word = word.lstrip('-') if carried else word
        held = word.endswith('_')
        word = word.rstrip('_')
        more = word.endswith('-') and len(word) > 1
        word = word.rstrip('-') if more else word
        if not word:
            continue
        inside = going or carried
        syllabic = ('middle' if more else 'end') if inside else ('begin' if more else 'single')
        out.append((word, syllabic, held))
        going = more
    return out


def attach(bars, verses: list[str]) -> int:
    """Put each verse's syllables under the notes of [bars] marked 'ly',
    in order. Returns how many marked notes went without a syllable in
    verse 1, which a check can report."""
    marked = [n for bar in bars for n in bar.notes if 'ly' in n.get('marks', [])]
    missing = 0
    for v, line in enumerate(verses, start=1):
        sung = syllables(line)
        for n, syl in zip(marked, sung):
            n.setdefault('lyrics', []).append((v, *syl))
        if v == 1:
            missing = max(0, len(marked) - len(sung))
    return missing
