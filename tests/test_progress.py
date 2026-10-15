"""A long job that goes quiet is flagged in `staffreader progress`: a job stuck
waiting on memory or a disk looks alive and says nothing."""

import time

from staffreader import progress


def _job(quiet_minutes: float, every_minutes: float) -> dict:
    now = time.time()
    hist = [(now - quiet_minutes * 60 - k * every_minutes * 60, 10 - k) for k in (3, 2, 1, 0)]
    return {'updated': now - quiet_minutes * 60, 'history': hist}


def test_a_job_quiet_far_longer_than_usual_is_stalled():
    assert progress._stalled(_job(quiet_minutes=75, every_minutes=4))


def test_a_job_whose_stage_is_working_is_not_stalled():
    import time
    assert progress._stalled(_job(quiet_minutes=75, every_minutes=4), heard=time.time() - 30) is None


def test_a_job_with_slow_steps_is_not_stalled_for_one_slow_step():
    assert progress._stalled(_job(quiet_minutes=12, every_minutes=6)) is None


def test_a_job_quiet_a_few_minutes_is_not_stalled():
    assert progress._stalled(_job(quiet_minutes=3, every_minutes=0.5)) is None
