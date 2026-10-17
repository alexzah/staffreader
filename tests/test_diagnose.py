"""The scorecard's figures."""

from staffreader import diagnose


def test_a_mark_between_a_parts_staves_counts_once_whichever_staff_reads_it():
    upper = 'clef:G2 key:0 time:4/4 hp< N:C5 d:whole hp. |'.split()
    lower = 'clef:F4 key:0 time:4/4 hp< N:C3 d:whole hp. |'.split()  # the answer labels it on both staves
    bare = 'clef:F4 key:0 time:4/4 N:C3 d:whole |'.split()  # the reading has it on the upper staff only
    records = [{'path': 'p0001-s0-1-1.png', 'marks_truth': upper, 'marks_read': upper},
               {'path': 'p0001-s0-1-2.png', 'marks_truth': lower, 'marks_read': bare}]
    assert diagnose.part_marks(records) == {'hairpins': 1.0}


def test_a_control_read_with_another_decoder_is_stale(tmp_path):
    import json
    from staffreader import evaluate
    from staffreader.baselines import staffreader as ours
    (tmp_path / 'ladder').mkdir()
    (tmp_path / 'ladder' / 'results.json').write_text(json.dumps([{'reader': 'staffreader-x-1-an-older-decoder'}]))
    why = evaluate.stale(tmp_path)
    assert any('another decoder' in w for w in why) and any('no records' in w for w in why)
    (tmp_path / 'ladder' / 'results.json').write_text(json.dumps([{'reader': f'staffreader-x-1-{ours.DECODER}'}]))
    assert not any('another decoder' in w for w in evaluate.stale(tmp_path))
