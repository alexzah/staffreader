"""Bars read too long, read again: triplets printed without their "3", and
a repeated figure written once too often."""

from fractions import Fraction

from staffreader import model


def notes(*pitches_and_lengths):
    out = []
    for pitch, length in pitches_and_lengths:
        out += [f'N:{pitch}', f'd:{length}']
    return out


def test_a_run_of_three_eighths_is_offered_as_a_triplet():
    bar = notes(('C5', 'eighth'), ('C5', 'eighth'), ('C5', 'eighth'), ('D5', 'quarter'), ('E5', 'quarter')) + ['|']
    options = model.triplet_options(bar, Fraction(3))
    assert [t for t in options[0] if t.startswith('tup:')] == ['tup:3:2'] * 3


def test_a_figure_written_once_too_often_is_offered_without_it():
    bar = notes(*[('E5', '16th'), ('E4', '16th')] * 7) + ['|']  # fourteen sixteenths in 3/4
    options = model.repeat_options(bar, Fraction(3))
    assert options == [notes(*[('E5', '16th'), ('E4', '16th')] * 6) + ['|']]


def test_a_bar_that_fills_its_time_is_never_offered_another_reading():
    bar = notes(('E5', '16th'), ('E4', '16th')) * 6 + ['|']
    assert model.repeat_options(bar, Fraction(3)) == []


def test_one_note_too_many_is_left_to_the_triplet_reading():
    bar = notes(('C5', 'eighth'), ('C5', 'eighth'), ('C5', 'eighth'), ('D5', 'quarter'), ('E5', 'quarter')) + ['|']
    assert model.repeat_options(bar, Fraction(3)) == []


def test_a_bar_that_opens_with_a_new_time_is_measured_against_it():
    reading = ['time:3/4'] + notes(('D4', 'half')) + ['.', '|'] + ['time:4/4'] + notes(*[('C4', 'eighth')] * 8) + ['|']
    spans = model.bar_spans(reading)
    assert [sp.limit for sp in spans] == [Fraction(3), Fraction(4)]
    assert all(max(sp.voices) == sp.limit for sp in spans)
