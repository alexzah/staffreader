"""Names a reviewer gives for each system's staves stand in for the labels
the text reader reads."""

from types import SimpleNamespace

import pytest

from staffreader.baselines import staffreader as ours


def _layout(*counts):
    staves, systems = 0, []
    for n in counts:
        systems.append(SimpleNamespace(staves=list(range(staves, staves + n)), parts=None))
        staves += n
    return SimpleNamespace(systems=systems)


def test_a_system_that_leaves_out_resting_parts_keeps_each_staff_on_its_part():
    lay = _layout(4, 2, 3)
    plan, parts = ours._given_plan(lay, [['Flute', 'Oboe', 'Violin 1', 'Viola'],
                                         ['Violin 1', 'Viola'],
                                         ['Flute', 'Violin 1', 'Cello']])
    assert [p.key for p in parts] == ['Flute', 'Oboe', 'Violin 1', 'Viola', 'Cello']
    assert plan == [[0, 1, 2, 3], [2, 3], [0, 2, 4]]


def test_neighbouring_staves_with_one_name_are_one_part():
    lay = _layout(4)
    plan, parts = ours._given_plan(lay, [['Voice', 'Piano', 'Piano', 'Cello']])
    assert plan == [[0, 1, 2]]
    assert lay.systems[0].parts == [[0], [1, 2], [3]]


def test_a_named_instrument_keeps_its_transposition():
    _, parts = ours._given_plan(_layout(2), [['Clarinetto in A', 'Corni in F']])
    assert [p.transpose for p in parts] == [-3, -7]


def test_a_count_that_does_not_match_the_page_says_so():
    with pytest.raises(ValueError, match='3 names given for 2 staves'):
        ours._given_plan(_layout(2), [['Flute', 'Oboe', 'Clarinet']])
