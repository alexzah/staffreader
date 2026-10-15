"""The layout stage finds every staff, system and barline exactly, on
pages MuseScore engraved, whose answer is exact, and on real scans, whose
answer was counted against the page by eye."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import needs
from staffreader import cases, layout, layout_suite, layout_truth

ALL = cases.all_cases()


def _page(case_id, track_name, k=0):
    case = next(c for c in ALL if c.id == case_id)
    track = next(t for t in case.tracks if t.name == track_name)
    needs.scan(track)
    image, page = case.pages(track)[k]
    return case, track, layout_suite.gray_page(image, page)


@needs.keys
@needs.musescore
@pytest.mark.parametrize('angle', [0.0, 0.3, -0.45, 1.2])
def test_a_tilted_page_is_turned_level(angle):
    _, _, gray = _page('lark-opening', 'synthetic-clean')
    h, w = gray.shape
    tilted = cv2.warpAffine(gray, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), (w, h), borderValue=255)
    _, found = layout.straighten(layout.binarize(tilted))
    assert abs(found + angle) < 0.05


# Pages the layout is known to miss, and why. Each passes once fixed, and
# the test then says so.
KNOWN: dict[tuple[str, str], str] = {}



@needs.keys
@needs.musescore
@pytest.mark.parametrize('depth', [6.0, 10.0])
def test_a_page_bent_toward_its_foot_is_flattened(depth):
    """Paper that curls near the bottom of a scan tilts the last staves
    while the rest lie level. The page is bent here as such scans are,
    more toward the foot and opposite at either edge, and its layout must
    match the flat page's."""
    _, _, gray = _page('dissonance-opening', 'synthetic-clean')
    flat = layout.read(gray)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shift = depth * (yy / h) ** 2 * (2 * xx / w - 1)
    bent = cv2.remap(gray, xx, yy + shift, cv2.INTER_LINEAR, borderValue=255)
    found = layout.read(bent)
    assert found.bend is not None
    assert [[len(s.staves), s.measures] for s in found.systems] == [[len(s.staves), s.measures] for s in flat.systems]


def _tracks():
    for case in ALL:
        for track in case.tracks:
            why = KNOWN.get((case.id, track.name))
            marks = [pytest.mark.xfail(reason=why, strict=True)] if why else []
            yield pytest.param(case.id, track.name, id=f'{case.id}-{track.name}', marks=marks)


@needs.keys
@needs.musescore
@pytest.mark.slow
@pytest.mark.parametrize('case_id,track_name', list(_tracks()))
def test_every_page_s_layout_is_found_exactly(case_id, track_name):
    case = next(c for c in ALL if c.id == case_id)
    track = next(t for t in case.tracks if t.name == track_name)
    needs.scan(track)
    for k, (image, page) in enumerate(case.pages(track)):
        truth = case.layout_truth(track, k)
        found = layout.read(layout_suite.gray_page(image, page))
        if isinstance(truth, layout.Layout):
            s = layout_truth.compare(truth, found).summary()
            assert s['line_error_spaces'] < 0.2
        else:
            s = layout_truth.compare_counted(truth, found).summary()
        assert (s['staves_f1'], s['systems_exact'], s['barlines_f1'], s['bars_per_system_exact']) == (1, 1, 1, 1), s


def test_a_ledger_line_never_stands_in_for_a_staff_line():
    """Five lines a staff space apart, the top one short as a ledger line
    is, and the four below it long, with a long fifth beneath: the staff
    is the five long lines."""
    ink = np.zeros((400, 1200), np.uint8)
    for y in (100, 120, 140, 160, 180):
        ink[y:y + 2, 100:1100] = 1
    ink[80:82, 500:540] = 1  # a ledger line above the staff
    staves = layout.find_staves(ink, 2, 18)
    assert len(staves) == 1
    assert [round(y) for y in staves[0].lines] == [100, 120, 140, 160, 180]
