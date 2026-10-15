"""A staff's instrument is read from its label: an orchestral score
names its staves in several languages, abbreviates them after the first
system, and leaves out the staves of instruments with nothing to play."""

from staffreader import instruments as I


def test_a_label_names_its_instrument_number_and_transposition():
    cases = {'Fl.': ('flute', None, 0), '2.Viol.': ('violin', '2', 0), 'Klar. (B)': ('clarinet', None, -2),
             'Corni in F': ('horn', None, -7), 'K-Fag.': ('contrabassoon', None, -12), 'Br.': ('viola', None, 0),
             'Clarinetto I in A': ('clarinet', '1', -3), 'Trombe in D': ('trumpet', None, 2), 'Pauken': ('timpani', None, 0)}
    for label, (name, number, transpose) in cases.items():
        i = I.instrument(label)
        assert (i.name, i.number, i.transpose) == (name, number, transpose), label


def test_a_system_that_leaves_out_staves_keeps_each_on_its_part():
    parts, plan = I.assign([['Fl.', 'Ob.', 'Cl.', 'Fg.', 'Cor. (C)', 'Tr. (C)', 'Pk.', 'Vl.', 'Vl.', 'Vla.', 'Vc. e B.'],
                            ['Ob.', 'Fg.', 'Vl.', 'Vl.', 'Vla.', 'Vc. e B.'],
                            ['Ob.', None, 'Vl.', None, 'Vla.', 'Vc. e B.']])
    assert len(parts) == 11
    assert plan[1] == [1, 3, 7, 8, 9, 10]
    assert plan[2] == [1, 3, 7, 8, 9, 10]  # an unread staff takes the part the system before showed there
