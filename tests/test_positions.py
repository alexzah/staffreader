"""A reading's notes matched to the noteheads found on the page."""

from types import SimpleNamespace

from staffreader import positions, tokens


def test_a_pitch_stands_at_its_height_on_the_staff():
    assert positions.step('E4', 'G2') == 0 and positions.step('F5', 'G2') == 8
    assert positions.step('G2', 'F4') == 0 and positions.step('C4', 'C3') == 4
    assert positions.step('E3', 'G2-8') == 0


def test_each_onset_takes_the_column_of_heads_at_its_height():
    # A treble staff with lines 10 pixels apart, bottom line at y = 100.
    staff = SimpleNamespace(space=10.0, bottom=100.0, x0=0.0)
    y = lambda s: 100 - 5 * s  # noqa: E731, a step is half a space
    found = [(30, y(4)), (45, y(2)), (60, y(2)), (60, y(4)), (90, y(6))]  # the head at 45 is an accidental's dot
    bars = tokens.parse('clef:G2 key:0 time:3/4 N:B4 d:quarter N:G4 d:quarter + N:B4 d:quarter N:D5 d:quarter |'.split())
    assert positions.place(bars, staff, found, [120.0]) == 4
    assert [(n['pitch'], n['x']) for n in bars[0].notes] == [('B4', 30), ('G4', 60), ('B4', 60), ('D5', 90)]
