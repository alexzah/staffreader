"""The harness is trusted only if it charges exactly the errors a reader
makes. Each test injects one kind of error Audiveris made on the Ricordi
Boccherini page into a score we hold, and checks the metric moves where
it should and nowhere else."""

from __future__ import annotations

import copy
import pathlib
import subprocess
import xml.etree.ElementTree as ET

import pytest

import needs
from staffreader import answers, canonical, metrics, render

BOCCHERINI = answers.FOLDER / 'boccherini-minuet-p22.musicxml'
PETZOLD = answers.FOLDER / 'petzold-minuet.musicxml'
EMPEROR = answers.FOLDER / 'emperor-quartet-2.musicxml'

needs_scores = pytest.mark.skipif(not BOCCHERINI.exists(), reason='the answer keys are not fetched (staffreader fetch)')


def tree(path) -> ET.Element:
    return ET.fromstring(canonical.load(path))


def score(root: ET.Element) -> canonical.Score:
    return canonical.parse(ET.tostring(root))


def summary(truth: ET.Element, reading: ET.Element) -> dict:
    return metrics.compare(score(truth), score(reading)).summary()


@needs_scores
@pytest.mark.parametrize('path', [BOCCHERINI, PETZOLD, EMPEROR])
def test_a_score_against_itself_is_perfect(path):
    s = summary(tree(path), tree(path))
    for k in ('note_f1_core', 'note_f1_rhythm', 'note_f1_full', 'rest_f1', 'bar_exact_core', 'bar_exact_full',
              'attributes', 'structure'):
        assert s[k] == 1.0, k
    assert s['measures_missed'] == s['measures_extra'] == 0


@needs_scores
def test_a_missing_bar_costs_one_bar_not_every_bar_after_it():
    truth = tree(PETZOLD)
    reading = copy.deepcopy(truth)
    part = reading.find('part')
    part.remove(part.findall('measure')[5])
    report = metrics.compare(score(truth), score(reading))
    s = report.summary()
    assert s['measures_missed'] == 1 and s['measures_extra'] == 0
    assert sum(not b.exact_full for b in report.bars) == 1


@needs_scores
def test_an_extra_bar_is_counted_as_extra():
    truth = tree(PETZOLD)
    reading = copy.deepcopy(truth)
    part = reading.find('part')
    part.insert(list(part).index(part.findall('measure')[3]), copy.deepcopy(part.findall('measure')[3]))
    s = summary(truth, reading)
    assert s['measures_extra'] == 1 and s['measures_missed'] == 0
    assert s['bar_exact_full'] == 1.0


def _notes(root, part_index=0):
    return [n for n in root.findall('part')[part_index].iter('note') if n.find('rest') is None]


@needs_scores
def test_a_sharp_read_for_a_natural_is_one_wrong_note():
    truth = tree(BOCCHERINI)
    reading = copy.deepcopy(truth)
    note = _notes(reading, 3)[10]
    p = note.find('pitch')
    alter = p.find('alter')
    if alter is None:
        alter = ET.SubElement(p, 'alter')
        p.remove(alter)
        p.insert(1, alter)
    alter.text = str(int(float(alter.text or 0)) + 1)
    report = metrics.compare(score(truth), score(reading))
    assert report.notes['core'].fn == 1 and report.notes['core'].fp == 1
    assert sum(not b.exact_core for b in report.bars) == 1
    assert report.errors == {'wrong accidental': 1}


@needs_scores
def test_a_lost_key_signature_costs_the_key_and_the_notes_it_alters():
    truth = tree(BOCCHERINI)
    reading = copy.deepcopy(truth)
    viola = reading.findall('part')[2]
    fifths = [k for k in viola.iter('fifths')]
    assert fifths, 'the viola part should carry a key'
    sharps = {'F', 'C', 'G', 'D'}  # E major: four sharps
    for f in fifths:
        f.text = '0'
    changed = 0
    for n in viola.iter('note'):
        p = n.find('pitch')
        if p is not None and p.findtext('step') in sharps and n.find('accidental') is None \
                and p.findtext('alter') == '1':
            p.remove(p.find('alter'))
            changed += 1
    report = metrics.compare(score(truth), score(reading))
    assert report.summary()['attributes'] < 1.0
    assert report.notes['core'].fn == changed
    assert report.summary()['structure'] == 1.0


@needs_scores
def test_dropped_eighth_rests_cost_rests_and_not_notes():
    truth = tree(BOCCHERINI)
    reading = copy.deepcopy(truth)
    dropped = 0
    for part in reading.findall('part'):
        for m in part.findall('measure'):
            for n in list(m):
                if n.tag == 'note' and n.find('rest') is not None and n.findtext('type') == 'eighth':
                    # A reader that misses a rest still places what follows.
                    forward = ET.Element('forward')
                    ET.SubElement(forward, 'duration').text = n.findtext('duration')
                    m.insert(list(m).index(n), forward)
                    m.remove(n)
                    dropped += 1
    assert dropped
    report = metrics.compare(score(truth), score(reading))
    assert report.rests.fn == dropped
    assert report.summary()['note_f1_core'] == 1.0


@needs_scores
def test_a_grace_note_read_as_a_full_note_is_charged_to_its_bar():
    truth = tree(BOCCHERINI)
    reading = copy.deepcopy(truth)
    for part in reading.findall('part'):
        for m in part.findall('measure'):
            for n in m.findall('note'):
                if n.find('grace') is not None:
                    n.remove(n.find('grace'))
                    ET.SubElement(n, 'duration').text = n.findtext('../attributes/divisions') or '1'
                    report = metrics.compare(score(truth), score(reading))
                    assert report.notes['core'].fn >= 1 and report.notes['core'].fp >= 1
                    assert sum(not b.exact_core for b in report.bars) >= 1
                    return
    pytest.fail('no grace note on the page')


@needs_scores
def test_a_repeat_read_as_a_plain_barline_costs_only_structure():
    truth = tree(BOCCHERINI)
    reading = copy.deepcopy(truth)
    removed = 0
    for b in reading.iter('barline'):
        r = b.find('repeat')
        if r is not None:
            b.remove(r)
            removed += 1
    assert removed
    s = summary(truth, reading)
    assert s['structure'] < 1.0
    assert s['note_f1_full'] == 1.0 and s['bar_exact_core'] == 1.0


@needs_scores
def test_parts_read_in_another_order_are_matched_by_their_music():
    truth = tree(EMPEROR)
    reading = copy.deepcopy(truth)
    parts = reading.findall('part')
    for p in parts:
        reading.remove(p)
    for p in reversed(parts):
        reading.append(p)
    s = summary(truth, reading)
    assert s['note_f1_full'] == 1.0


@needs.musescore
@needs_scores
@pytest.mark.slow
def test_musescore_s_own_export_reads_the_same(tmp_path_factory):
    """MuseScore re-exports a score with its own beaming, voices and
    divisions; what it means must not change."""
    work = render.CACHE / 'tests' / 'reexport'
    x = render.excerpt(PETZOLD, 1, 32, work / 'petzold.musicxml')
    out = work / 'petzold-again.musicxml'
    render.engrave(x, work / 'petzold.pdf')  # proves MuseScore runs here
    home = render.CACHE / 'mscore-home'
    subprocess.run([render.MSCORE, '-o', str(out), str(x)], capture_output=True, timeout=600,
                   env={**__import__('os').environ, 'HOME': str(home), 'QT_QPA_PLATFORM': 'offscreen'})
    s = metrics.compare(canonical.read(x), canonical.read(out)).summary()
    assert s['note_f1_core'] >= 0.999
    assert s['bar_exact_core'] >= 0.99


@needs_scores
def test_a_missing_time_signature_is_one_correction_not_one_per_bar():
    truth = tree(PETZOLD)
    reading = copy.deepcopy(truth)
    for attributes in reading.iter('attributes'):
        for t in attributes.findall('time'):
            attributes.remove(t)
    report = metrics.compare(score(truth), score(reading))
    assert sum(not b.exact_full for b in report.bars) == 1
    assert report.summary()['attributes'] < 1.0


def test_a_page_turn_prints_no_change_where_nothing_changes():
    """Each page is read on its own and opens with its clef and key, as
    every system does. Joined, a page's first bar changes only what
    differs from the page before."""
    from fractions import Fraction
    from staffreader import suite

    def bar(key, clef='G2', changes=('clef1', 'key')):
        return canonical.Measure(0, '1', False, [canonical.Event('rest', 1, Fraction(0), Fraction(4))],
                                 clefs={1: clef}, key=key, time=(4, 4), changes=frozenset(changes))

    def page(*bars):
        return canonical.Score([canonical.Part('P1', 'Violin', 1, list(bars))])

    joined = suite.join([page(bar(2, changes=('clef1', 'key', 'time')), bar(2, changes=())),
                         page(bar(2)), page(bar(-1)), page(bar(-1, clef='C3'))])
    assert [sorted(m.changes) for m in joined.parts[0].measures] == [
        ['clef1', 'key', 'time'], [], [], ['key'], ['clef1']]


@needs_scores
def test_a_part_printed_on_another_s_staff_leaves_the_scan_s_answer():
    """Mozart 40's edition prints the bass on the cello's staff; the
    transcription writes it apart. The scan's answer leaves it out, and
    the engraving's keeps it."""
    from staffreader import cases
    case = next(c for c in cases.all_cases() if c.id == 'mozart-40-opening')
    scan = next(t for t in case.tracks if t.kind == 'scan')
    engraved = next(t for t in case.tracks if t.kind == 'synthetic')
    assert 'Contrabass' not in [p.name for p in case.truth(scan).parts]
    assert 'Contrabass' in [p.name for p in case.truth(engraved).parts]


MARKED = ['lied-beethoven-molly', 'boccherini-p22', 'mozart-40-p2', 'dissonance-opening', 'lied-brahms-p141',
          'arabesque-1-p3']


@needs_scores
@pytest.mark.parametrize('case_id', MARKED)
def test_every_kind_of_mark_matches_itself(case_id):
    from staffreader import cases
    truth = next(c for c in cases.all_cases() if c.id == case_id).truth()
    marks = metrics.compare(truth, truth).summary()['marks']
    assert marks and all(v['f1'] == 1.0 for v in marks.values()), marks


@needs.musescore
@needs_scores
@pytest.mark.parametrize('case_id', MARKED)
def test_musescore_s_own_export_keeps_every_mark(case_id):
    """Lyrics in five verses, slurs, hairpins, dynamics, fermatas on
    chords and the rest read back from MuseScore's export as they were
    written, from excerpts that cut through slurs and attribute changes."""
    import os
    from staffreader import cases
    case = next(c for c in cases.all_cases() if c.id == case_id)
    work = render.CACHE / 'tests' / 'reexport-marks'
    x = render.excerpt(case.truth_file, case.first, case.last, work / f'{case_id}.musicxml')
    out = work / f'{case_id}-again.musicxml'
    out.unlink(missing_ok=True)
    subprocess.run([render.MSCORE, '-o', str(out), str(x)], capture_output=True, timeout=600,
                   env={**os.environ, 'HOME': str(render.CACHE / 'mscore-home'), 'QT_QPA_PLATFORM': 'offscreen'})
    s = metrics.compare(canonical.read(x), canonical.read(out)).summary(case.masks)  # the families the case scores
    assert s['note_f1_core'] == 1.0
    assert all(v['f1'] >= 0.999 for v in s['marks'].values()), {f: v for f, v in s['marks'].items() if v['f1'] < 0.999}
