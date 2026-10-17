"""Directions in words found on the page, put on the staves they belong to."""

from types import SimpleNamespace

from staffreader import tokens, words


def test_a_found_word_goes_to_the_note_it_stands_at_and_replaces_the_spelled_one():
    staff = SimpleNamespace(top=100, bottom=140, x0=0, space=10)
    bars = [tokens.parse('clef:G2 key:0 time:4/4 w( c:a c:r c:c c:o ) N:C5 d:half N:D5 d:half | N:E5 d:whole |'.split())]
    for n, x in zip(bars[0][0].notes, (30, 80)):
        n['x'] = x
    placed = words.place_words([((75, 150, 120, 165), 'dolce'), ((10, 80, 60, 95), 'a 2.'), ((30, 150, 40, 160), 'p')],
                               [staff], bars, [120.0, 200.0])
    assert placed == 2  # the dynamic is the staff reader's to read
    texts = [''.join(t[2:] for t in n.get('before', []) if t.startswith('c:')) for n in bars[0][0].notes]
    assert texts == ['a␣2.', 'dolce']
