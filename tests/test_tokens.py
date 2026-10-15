"""The reader's tokens hold everything a staff prints: every bar the page
shows, written as tokens a system at a time and read back, is the same
music."""

from __future__ import annotations

import collections
from fractions import Fraction
import xml.etree.ElementTree as ET

import pytest

from staffreader import canonical, cases, data, tokens

have_scores = pytest.mark.skipif(not (data.BUNDLES and data.BUNDLES.exists()), reason='score bundles are not on this machine')

# Bundles whose MusicXML contradicts itself: Dissonance II-IV print sharps
# on natural pitches, and a few files leave ties open or nest tuplets.
FLAWED = {'dissonance-quartet-2', 'dissonance-quartet-3', 'dissonance-quartet-4'}


def _look(e: canonical.Event):
    return (e.kind, e.onset, e.duration, e.pitch, e.tie_start, e.tie_stop, e.accidental)


def round_trip(path, per_stretch: int = 4) -> collections.Counter:
    """How many of a score's staff-bars come back exactly, how many do
    not, and how many the page cannot show."""
    root = ET.fromstring(canonical.load(path))
    truth = canonical.read(path)
    staves_of_parts, readable = [], []
    for part in root.findall('part'):
        count = max([int(s.text) for s in part.iter('staves')] or [1])
        staves, ok = [], []
        for s in range(1, count + 1):
            bars, states = tokens.from_part(part, s)
            read = []
            for a in range(0, len(bars), per_stretch):
                stretch = tokens.stretch(bars, states, a, min(a + per_stretch - 1, len(bars) - 1))
                assert all(t in VOCABULARY or t == '?' for t in stretch.tokens), stretch.tokens
                read += tokens.parse(stretch.tokens)
            staves.append(read)
            ok.append([st['readable'] for st in states])
        staves_of_parts.append(staves)
        readable.append(ok)
    ours = canonical.parse(tokens.to_musicxml(staves_of_parts))
    out = collections.Counter()
    for p, (pt, po) in enumerate(zip(truth.parts, ours.parts)):
        for s, ok in enumerate(readable[p]):
            for i, r in enumerate(ok):
                if not r:
                    out['unreadable'] += 1
                    continue
                same = [_look(e) for e in pt.measures[i].events if e.staff == s + 1] == \
                    [_look(e) for e in po.measures[i].events if e.staff == s + 1]
                out['exact' if same else 'wrong'] += 1
    return out


VOCABULARY = set(tokens.vocabulary())


def test_the_vocabulary_names_each_token_once():
    words = tokens.vocabulary()
    assert len(words) == len(set(words))


@have_scores
@pytest.mark.parametrize('bundle', ['paganini-caprice-24', 'bach-air-2', 'boccherini-minuet-p22',
                                    'eine-kleine-nachtmusik-1', 'bach-partita-2-4', 'dissonance-quartet-1',
                                    'emperor-quartet-2', 'lark-quartet-1', 'petzold-minuet'])
def test_a_ladder_score_comes_back_bar_for_bar(bundle):
    got = round_trip(data.BUNDLES / bundle / 'score.musicxml')
    assert got['wrong'] <= 0.01 * got['exact'], got


@have_scores
@pytest.mark.slow
def test_nearly_every_bar_the_corpus_prints_comes_back():
    total = collections.Counter()
    for d in sorted(data.BUNDLES.iterdir()):
        if (d / 'score.musicxml').exists() and d.name not in FLAWED:
            total += round_trip(d / 'score.musicxml')
    # Voices admit the hardest keyboard bars, two voices and spaces, so a
    # few more of them miss than when only one-voice bars were counted.
    assert total['wrong'] <= 0.003 * total['exact'], total


def _bars(xml: str, staff: int = 1):
    part = ET.fromstring(xml).find('part')
    return tokens.from_part(part, staff)


SCORE = """<score-partwise><part id="P1">
<measure number="1"><attributes><divisions>12</divisions><key><fifths>-1</fifths></key>
  <time><beats>2</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
  <note><pitch><step>F</step><alter>1</alter><octave>5</octave></pitch><duration>12</duration><tie type="start"/>
    <type>quarter</type><accidental>sharp</accidental></note>
  <note><pitch><step>F</step><alter>1</alter><octave>5</octave></pitch><duration>12</duration><type>quarter</type></note>
  <barline location="right"><bar-style>light-heavy</bar-style><repeat direction="backward"/></barline></measure>
<measure number="2">
  <note><pitch><step>F</step><alter>1</alter><octave>5</octave></pitch><duration>12</duration><tie type="stop"/>
    <type>quarter</type></note>
  <note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch><duration>4</duration>
    <type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
  <note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch><duration>5</duration>
    <type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
  <note><rest/><duration>3</duration><type>16th</type></note>
</measure></part></score-partwise>"""


def test_a_staff_is_written_as_the_page_prints_it():
    bars, states = _bars(SCORE)
    assert tokens.stretch(bars, states, 0, 1).tokens == [
        'clef:G2', 'key:-1', 'time:2/4',
        'acc:sharp', 'N:F5', 'd:quarter', '~', 'N:F5', 'd:quarter', ':|',
        'N:F5', 'd:quarter', 'N:B4', 'd:eighth', 'tup:3:2', 'R', 'd:16th', '|']
    assert tokens.stretch(bars, states, 1, 1).tokens[:2] == ['clef:G2', 'key:-1']
    # The second triplet eighth lasts longer than its look says, so the
    # page cannot show that bar.
    assert [s['readable'] for s in states] == [True, False]


def test_a_tie_carries_its_sharp_over_the_barline():
    bars = tokens.parse(['clef:G2', 'key:0', 'time:2/4', 'acc:sharp', 'N:F5', 'd:half', '~', '|',
                         'N:F5', 'd:quarter', 'N:F5', 'd:quarter', '|'])
    m = canonical.parse(tokens.to_musicxml([[bars]])).parts[0].measures[1]
    assert [(e.pitch, e.tie_stop) for e in m.events] == [('F#5', True), ('F5', False)]


def test_a_repeat_that_starts_the_next_bar_is_written_on_its_left():
    bars = tokens.parse(['clef:F4', 'key:2', 'time:3/4', 'R:measure', '|:', 'N:D3', 'd:half', '.', ':|'])
    part = canonical.parse(tokens.to_musicxml([[bars]])).parts[0]
    assert part.measures[0].clefs == {1: 'F4'}
    assert part.measures[0].events[0].duration == 3
    assert part.measures[1].barlines == {'repeat-forward', 'repeat-backward'}
    assert part.measures[1].events[0].pitch == 'D3'


def test_a_quintuplet_keeps_its_exact_length():
    bars = tokens.parse(['clef:G2', 'key:0', 'time:1/4'] + ['N:C5', 'd:16th', 'tup:5:4'] * 5 + ['|'])
    events = canonical.parse(tokens.to_musicxml([[bars]])).parts[0].measures[0].events
    assert [e.onset for e in events] == [Fraction(k, 5) for k in range(5)]


TWO_VOICES = """<score-partwise><part id="P1">
<measure number="1"><attributes><divisions>4</divisions><key><fifths>0</fifths></key>
  <time><beats>2</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
  <note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>quarter</type></note>
  <note><pitch><step>F</step><alter>1</alter><octave>5</octave></pitch><duration>4</duration><tie type="start"/><voice>1</voice>
    <type>quarter</type><accidental>sharp</accidental></note>
  <backup><duration>8</duration></backup>
  <forward><duration>2</duration><voice>2</voice></forward>
  <note><pitch><step>F</step><octave>5</octave></pitch><duration>4</duration><voice>2</voice><type>quarter</type></note>
  <note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration><voice>2</voice><type>eighth</type></note>
</measure>
<measure number="2">
  <note><pitch><step>F</step><alter>1</alter><octave>5</octave></pitch><duration>8</duration><tie type="stop"/><voice>1</voice><type>half</type></note>
</measure></part></score-partwise>"""


def test_two_voices_on_a_staff_are_written_one_after_the_other():
    bars, states = _bars(TWO_VOICES)
    assert bars[0] == ['N:E5', 'd:quarter', 'acc:sharp', 'N:F5', 'd:quarter', '~',
                       '&', 'S', 'd:eighth', 'N:F5', 'd:quarter', 'N:C5', 'd:eighth', '|']
    assert all(s['readable'] for s in states)


def test_voices_read_back_with_their_onsets_accidentals_and_ties():
    bars, states = _bars(TWO_VOICES)
    read = tokens.parse(tokens.stretch(bars, states, 0, 1).tokens)
    m1, m2 = canonical.parse(tokens.to_musicxml([[read]])).parts[0].measures
    got = {(str(e.onset), e.pitch, e.tie_start, e.tie_stop) for e in m1.events}
    # The sharp printed on beat 2 in voice 1 does not reach voice 2's F,
    # which sounds half a beat earlier; the tie stays in voice 1.
    assert got == {('0', 'E5', False, False), ('1', 'F#5', True, False),
                   ('1/2', 'F5', False, False), ('3/2', 'C5', False, False)}
    assert [(e.pitch, e.tie_stop) for e in m2.events] == [('F#5', True)]


def _marks_round_trip(path, per_stretch: int = 4) -> dict:
    """Every family of marks, written as tokens a system at a time, read
    back and written as MusicXML, against the score."""
    from staffreader import metrics
    root = ET.fromstring(canonical.load(path))
    truth = canonical.read(path)
    staves_of_parts = []
    for part in root.findall('part'):
        count = max([int(s.text) for s in part.iter('staves')] or [1])
        staves = []
        for s in range(1, count + 1):
            bars, states = tokens.from_part(part, s)
            read = []
            for a in range(0, len(bars), per_stretch):
                read += tokens.parse(tokens.stretch(bars, states, a, min(a + per_stretch - 1, len(bars) - 1)).tokens)
            staves.append(read)
        staves_of_parts.append(staves)
    ours = canonical.parse(tokens.to_musicxml(staves_of_parts))
    for a, b in zip(truth.parts, ours.parts):
        b.transpose = a.transpose  # read from the staff label, not the staff
    return metrics.compare(truth, ours).summary()['marks']


@have_scores
@pytest.mark.parametrize('bundle', ['boccherini-minuet-p22', 'lark-quartet-1', 'emperor-quartet-2', 'petzold-minuet',
                                    'arabesque-1'])
def test_every_mark_comes_back_through_the_tokens(bundle):
    """Slurs, hairpins, dynamics, words, articulations, ornaments,
    fermatas and endings come back where they were. Lyrics wait for the
    text reader."""
    marks = _marks_round_trip(data.BUNDLES / bundle / 'score.musicxml')
    assert all(v['f1'] >= 0.97 for f, v in marks.items() if f != 'lyrics'), marks


def test_marks_join_the_notes_of_a_notes_only_reading():
    """A notes-only reading keeps its notes and takes the marks of a
    marked reading of the same staff, even where that reading misread a
    rhythm and moved the onsets after it."""
    plain = tokens.parse('clef:G2 key:0 N:C5 d:quarter N:D5 d:quarter N:E5 d:half |'.split())
    marked = tokens.parse('clef:G2 key:0 dyn:p N:C5 d:quarter sl( m:staccato N:D5 d:eighth sl) N:E5 d:half m:fermata |'
                          .split())
    m = canonical.parse(tokens.to_musicxml([[tokens.merge_marks(plain, marked)]])).parts[0].measures[0]
    assert [(e.onset, e.pitch, e.marks) for e in m.events] == [
        (0, 'C5', ('slur-start', 'staccato')), (1, 'D5', ('slur-stop',)), (2, 'E5', ('fermata',))]
    assert m.directions == ((0, 'dyn:p'),)


def test_a_tenor_part_in_a_plain_treble_clef_is_written_an_octave_lower():
    from staffreader.baselines import staffreader as ours
    bars = tokens.parse('clef:G2 key:0 time:4/4 N:A4 d:half N:C5 d:half | N:G4 d:whole |'.split())
    shown = tokens.parse('clef:G2-8 key:0 time:4/4 N:A3 d:whole |'.split())
    ours.tenor_clef([bars, shown])
    assert bars[0].clef == 'G2-8'
    assert [n['pitch'] for b in bars for n in b.notes] == ['A3', 'C4', 'G3']
    assert [n['pitch'] for b in shown for n in b.notes] == ['A3']


THIRD_GROUP = """<score-partwise><part-list><score-part id="P1"><part-name>Harpsichord</part-name></score-part></part-list>
<part id="P1">
<measure number="1"><attributes><divisions>2</divisions><key><fifths>0</fifths></key>
  <time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves>
  <clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>
  <harmony><root><root-step>C</root-step></root><kind>major</kind>
    <degree><degree-value>9</degree-value><degree-alter>0</degree-alter><degree-type>add</degree-type></degree>
    <bass><bass-step>E</bass-step></bass><staff>1</staff></harmony>
  <note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>half</type>
    <notehead>mi</notehead><staff>1</staff>
    <notations><ornaments><schleifer/></ornaments><technical><string>2</string><fret>5</fret></technical></notations></note>
  <harmony><root><root-step>F</root-step><root-alter>1</root-alter></root><kind>half-diminished</kind><staff>1</staff></harmony>
  <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>half</type><staff>1</staff></note>
  <backup><duration>8</duration></backup>
  <figured-bass><figure><figure-number>6</figure-number></figure><figure><prefix>sharp</prefix><figure-number>4</figure-number><extend/></figure></figured-bass>
  <note><pitch><step>C</step><octave>3</octave></pitch><duration>8</duration><voice>5</voice><type>whole</type><staff>2</staff></note>
</measure>
<measure number="2"><attributes><measure-style><multiple-rest>3</multiple-rest></measure-style></attributes>
  <note><rest measure="yes"/><duration>8</duration><voice>1</voice><staff>1</staff></note><backup><duration>8</duration></backup>
  <note><rest measure="yes"/><duration>8</duration><voice>5</voice><staff>2</staff></note></measure>
<measure number="3"><note><rest measure="yes"/><duration>8</duration><voice>1</voice><staff>1</staff></note><backup><duration>8</duration></backup>
  <note><rest measure="yes"/><duration>8</duration><voice>5</voice><staff>2</staff></note></measure>
<measure number="4"><note><rest measure="yes"/><duration>8</duration><voice>1</voice><staff>1</staff></note><backup><duration>8</duration></backup>
  <note><rest measure="yes"/><duration>8</duration><voice>5</voice><staff>2</staff></note></measure>
<measure number="5"><note><pitch><step>G</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice><type>whole</type><staff>1</staff></note>
  <backup><duration>8</duration></backup>
  <note><pitch><step>G</step><octave>2</octave></pitch><duration>8</duration><voice>5</voice><type>whole</type><staff>2</staff></note></measure>
</part></score-partwise>"""


def test_the_third_group_of_marks_comes_back_through_the_tokens(tmp_path):
    """Chord symbols, figured bass, tablature, shape notes, the Schleifer
    and a multi-bar rest, written as tokens and read back, are the score's;
    the multi-bar rest prints as one bar with its count."""
    from staffreader import metrics
    path = tmp_path / 'third.musicxml'
    path.write_text(THIRD_GROUP)
    part = ET.fromstring(THIRD_GROUP).find('part')
    upper, states = tokens.from_part(part, 1)
    assert upper[1] == ['R:multi', 'mr:3', '|'] and upper[2] == [] and upper[3] == []
    assert 'ch(' in upper[0] and 'm:schleifer' in upper[0] and 'head:mi' in upper[0] and 'fret:5' in upper[0]
    lower, _ = tokens.from_part(part, 2)
    assert 'fb(' in lower[0]
    truth = canonical.read(path)
    ours = canonical.parse(tokens.to_musicxml([[tokens.parse(tokens.stretch(b, s, 0, len(b) - 1).tokens)
                                                for b, s in (tokens.from_part(part, 1), tokens.from_part(part, 2))]]))
    summary = metrics.compare(truth, ours).summary()
    for family in ('chord symbols', 'figured bass', 'tablature', 'noteheads', 'ornaments', 'multi-bar rests'):
        assert summary['marks'][family]['f1'] == 1.0, (family, summary['marks'])
    assert summary['bar_exact_full'] == 1.0


def test_an_older_checkpoint_never_writes_a_token_it_never_learned():
    """A vocabulary grown since a checkpoint was trained gives the new
    tokens a zero output row and a low bias, so they never outscore one
    the checkpoint learned."""
    import torch
    from staffreader import model as M
    net = M.Reader()
    old = {k: v[:-20] if k in ('embed.weight', 'out.weight', 'out.bias') else v for k, v in net.state_dict().items()}
    fitted = M.grown(old, net)
    assert torch.all(fitted['out.weight'][-20:] == 0) and torch.all(fitted['out.bias'][-20:] == -8.0)


def test_an_older_checkpoint_reads_as_it_did(tmp_path):
    """Loaded into a grown vocabulary, a checkpoint neither writes nor
    weighs the tokens it never learned: their scores are minus infinity
    when it reads, so every probability is the one it was trained to give."""
    import torch
    from staffreader import model as M
    config = {'dim': 64, 'heads': 2, 'context_layers': 1, 'decoder_layers': 1}
    net = M.Reader(**config)
    old = {k: v[:-20] if k in ('embed.weight', 'out.weight', 'out.bias') else v for k, v in net.state_dict().items()}
    torch.save({'model': old, 'config': config}, tmp_path / 'old.pt')
    loaded = M.load(tmp_path / 'old.pt', device='cpu')
    assert loaded.learned == len(M.VOCAB) - 20
    memory = torch.zeros(1, 4, 64)
    out = loaded.logits(memory, torch.zeros(1, 4, dtype=torch.bool), torch.tensor([[M.BOS]]))
    assert torch.isinf(out[0, -1, -20:]).all() and torch.isfinite(out[0, -1, :-20]).all()
