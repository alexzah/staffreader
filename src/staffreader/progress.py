"""How far every long job has got, in one place.

The standard: no job of ten minutes or more runs without a
clear progress indicator that is easy to check. Each long job keeps
PROGRESS/<job>.json up to date: what it is counting, how many of how
many, when it started, and the latest word on how it is going.

    uv run staffreader progress

prints running jobs first, then any that stopped without finishing,
then those finished in the last FINISHED_HOURS. A job started inside
another (a training run inside an experiment, say) is shown indented
under it: it names its parent through STAFFREADER_PROGRESS_PARENT, or it
runs in the parent's process or one of its children. A training run is
read straight from its model's log."""

from __future__ import annotations

import datetime
import json
import os
import re
import time

from .render import CACHE

PROGRESS = CACHE / 'progress'
CHECK = 'uv run staffreader progress'
FINISHED_HOURS = 12
BAR = 20


class Progress:
    def __init__(self, job: str, total: float, unit: str, what: str = ''):
        self.job, self.total, self.unit, self.what = job, total, unit, what
        self.started = time.time()
        self.history: list[tuple[float, float]] = []  # recent (time, done), for the pace
        self.parent = os.environ.get('STAFFREADER_PROGRESS_PARENT')
        self.ancestors = _ancestors(os.getpid())  # read now, while every parent process lives
        PROGRESS.mkdir(parents=True, exist_ok=True)
        self.update(0)

    def update(self, done: float, detail: str = '', finished: bool = False):
        self.history = (self.history + [(time.time(), done)])[-4:]
        state = {'history': self.history, 'job': self.job, 'what': self.what, 'unit': self.unit, 'done': done, 'total': self.total,
                 'started': self.started, 'updated': time.time(), 'pid': os.getpid(),
                 'parent': self.parent, 'ancestors': self.ancestors, 'finished': finished, 'detail': detail}
        file = re.sub(r'[^A-Za-z0-9._-]+', '_', self.job)  # a job's name may hold a slash or a space
        tmp = PROGRESS / f'.{file}.json'
        tmp.write_text(json.dumps(state))
        tmp.replace(PROGRESS / f'{file}.json')

    def finish(self, detail: str = ''):
        self.update(self.total, detail, finished=True)


def parent_of_children(job: str):
    """Mark everything started from here on as part of [job]."""
    os.environ['STAFFREADER_PROGRESS_PARENT'] = job


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _ancestors(pid: int) -> list[int]:
    out = []
    for _ in range(20):
        try:
            stat = open(f'/proc/{pid}/stat').read()
            pid = int(stat.rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if pid <= 1:
            break
        out.append(pid)
    return out


def _clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds >= 3600:
        return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m'
    return f'{seconds // 60}m{seconds % 60:02d}s' if seconds >= 60 else f'{seconds}s'


def _at(stamp: float) -> str:
    when = datetime.datetime.fromtimestamp(stamp)
    return f'{when:%H:%M}' if when.date() == datetime.date.today() else f'{when:%b %d %H:%M}'


def _bar(frac: float) -> str:
    full = int(round(frac * BAR))
    return '█' * full + '·' * (BAR - full)


def _training() -> list[dict]:
    """Training runs read from their logs."""
    out = []
    for log in sorted((CACHE / 'models').glob('*/log.jsonl')):
        run = log.parent / 'run.json'
        if not run.exists():
            continue
        cfg = json.loads(run.read_text())
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        last = rows[-1] if rows else {'step': 0, 'minutes': 0.0}
        evals = [r for r in rows if 'token_acc' in r]
        loss = next((r['loss'] for r in reversed(rows) if 'loss' in r), None)
        detail = f"step {last['step']}" + (f", loss {loss:.3f}" if loss is not None else '')
        if evals:
            e = evals[-1]
            held = []
            if e.get('token_acc'):
                held.append(f"rendered {e['token_acc']:.1%} of tokens, {e['staff_exact']:.0%} of staves exact")
            if 'real_token_acc' in e:
                held.append(f"real {e['real_token_acc']:.1%} of tokens, {e['real_staff_exact']:.0%} of staves exact")
            if held:
                detail += f"; held out, at step {e['step']}: " + '; '.join(held)
        by_steps = bool(cfg.get('steps'))
        total = cfg['steps'] if by_steps else cfg['hours'] * 60
        done = last['step'] if by_steps else round(last['minutes'], 1)
        steps = [(cfg['started'] + r['minutes'] * 60, r['step'] if by_steps else r['minutes']) for r in rows if 'loss' in r]
        out.append({'history': steps[-4:], 'job': f"train {log.parent.name}", 'what': cfg.get('what', ''),
                    'unit': 'steps' if by_steps else 'minutes', 'done': done, 'total': total,
                    'started': cfg['started'], 'updated': log.stat().st_mtime, 'pid': cfg['pid'],
                    'parent': cfg.get('parent'), 'finished': done >= total or bool(last.get('done')),
                    'detail': detail})
    return out


def _states() -> list[dict]:
    states = [json.loads(f.read_text()) for f in sorted(PROGRESS.glob('*.json'))] if PROGRESS.exists() else []
    states += _training()
    for s in states:
        s['state'] = 'finished' if s['finished'] else ('running' if _alive(s['pid']) else 'stopped')
    return states


def _parents(states: list[dict]) -> dict[str, str]:
    """Each job's parent: the one it names, or else the oldest running job
    that shares its process or is one of its process's ancestors."""
    jobs = {s['job']: s for s in states}
    out = {}
    # A parent started before its child and was still at work when the
    # child began; when both run in one process, the older is the parent.
    candidates = sorted(states, key=lambda s: s['started'])
    for s in states:
        if s.get('parent') in jobs and s['parent'] != s['job']:
            out[s['job']] = s['parent']
            continue
        # A dead process has no ancestors to read, but a job run inside its
        # parent's own process still shares its process number.
        line = [s['pid']] + (s.get('ancestors') or (_ancestors(s['pid']) if s['state'] == 'running' else []))
        for p in candidates:
            if p is s or p['started'] > s['started'] or p['updated'] < s['started']:
                continue
            if p['pid'] in line and not (p['pid'] == s['pid'] and out.get(p['job']) == s['job']):
                out[s['job']] = p['job']
                break
    return out


STALL_MINUTES = 10  # no job goes this long without a word unless something is wrong


def _stalled(s: dict, heard: float = 0.0) -> str | None:
    """A warning when a running job has gone quiet: no progress for
    STALL_MINUTES and for four times its usual gap between updates. A job
    stuck waiting on memory or a disk looks alive and says nothing, as the
    orchestral pairing did for an hour on 2026-10-14. [heard] is the last
    word from any of its stages, since a job in stages is quiet while one
    of them works."""
    quiet = time.time() - max(s['updated'], heard)
    hist = s.get('history') or []
    gaps = [b[0] - a[0] for a, b in zip(hist, hist[1:]) if b[0] > a[0]]
    usual = sorted(gaps)[len(gaps) // 2] if gaps else 0
    if quiet < STALL_MINUTES * 60 or quiet < 4 * usual:
        return None
    return f'STALLED? no progress for {_clock(quiet)}' + (f' (usually every {_clock(usual)})' if usual else '')


def machine() -> str | None:
    """The machine's memory, and a warning when jobs are fighting over it:
    little left, swap in use, and processes waiting on memory much of the
    time."""
    try:
        info = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo') if len(l.split()) >= 2}
        some = float(open('/proc/pressure/memory').readline().split()[2].split('=')[1])  # avg60
    except (OSError, ValueError, IndexError):
        return None
    gb = lambda kb: kb / 1024 ** 2
    free, total = gb(info.get('MemAvailable', 0)), gb(info.get('MemTotal', 1))
    swap = gb(info.get('SwapTotal', 0) - info.get('SwapFree', 0))
    line = f'Memory: {free:.0f} of {total:.0f} GB free, {swap:.0f} GB swapped, waiting on memory {some:.0f}% of the last minute'
    if some >= 20 or free < 0.1 * total:
        line += '\nWARNING: jobs are short of memory and may stall; stop one before it thrashes'
    return line


def _line(s: dict, depth: int, last: bool = True, heard: float = 0.0) -> list[str]:
    pad = '    ' * depth + (('└─ ' if last else '├─ ') if depth else '')
    frac = min(1.0, s['done'] / s['total']) if s['total'] else 0.0
    elapsed = (time.time() if s['state'] == 'running' else s['updated']) - s['started']
    count = f"{s['done']:g}/{s['total']:g} {s['unit']}"
    if s['state'] == 'running':
        # Time left at the recent pace, where there is one: a run's first
        # minutes (loading, compiling) are slower than the rest. Otherwise
        # at the pace so far, counted to now.
        hist = s.get('history') or []
        # The pace needs half a minute behind it: a pool of workers may hand
        # back several results at once.
        if len(hist) >= 3 and hist[-1][1] > hist[0][1] and hist[-1][0] - hist[0][0] >= 30:
            pace = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])
            left = (s['total'] - s['done']) / pace
        else:
            left = elapsed / frac - elapsed if frac > 0.01 else None
        when = (f"{_clock(left)} left, done ~{_at(time.time() + left)}" if left is not None
                else 'estimating time left')
        stalled = _stalled(s, heard)
        if stalled:
            when = stalled
        head = f"{pad}{s['job']}  [{_bar(frac)}] {frac:4.0%}  {count}  ·  {_clock(elapsed)} in, {when}"
    elif s['state'] == 'stopped':
        head = f"{pad}{s['job']}  stopped at {frac:.0%} ({count}), {_at(s['updated'])}: its process is gone"
    else:
        head = f"{pad}{s['job']}  finished {_at(s['updated'])} in {_clock(elapsed)}"
    lines = [head]
    note = s.get('detail') or s.get('what')
    if note:
        lines.append(' ' * len(pad) + '   ' + note)
    return lines


def report() -> str:
    states = _states()
    recent = time.time() - FINISHED_HOURS * 3600
    states = [s for s in states if s['state'] == 'running' or s['updated'] >= recent]
    if not states:
        return f'No jobs in the last {FINISHED_HOURS} hours.'
    parents = _parents(states)
    children: dict[str, list[dict]] = {}
    for s in states:
        if s['job'] in parents:
            children.setdefault(parents[s['job']], []).append(s)

    def latest(s) -> float:
        return max([s['updated']] + [latest(c) for c in children.get(s['job'], []) if c['state'] == 'running'])

    def tree(s, depth, last=True):
        out = _line(s, depth, last, latest(s))
        kids = sorted(children.get(s['job'], []), key=lambda c: c['started'])
        for k, c in enumerate(kids):
            out += tree(c, depth + 1, k == len(kids) - 1)
        return out

    out = [f'Jobs at {datetime.datetime.now():%H:%M}']
    if (m := machine()):
        out.append(m)
    out.append('')
    for title, which in (('RUNNING', 'running'), ('STOPPED (did not finish)', 'stopped'),
                         (f'FINISHED (last {FINISHED_HOURS} hours)', 'finished')):
        tops = [s for s in states if s['state'] == which and s['job'] not in parents]
        if which == 'finished':
            tops.sort(key=lambda s: -s['updated'])
        else:
            tops.sort(key=lambda s: s['started'])
        if not tops:
            continue
        out.append(f'{title}')
        for s in tops:
            out += tree(s, 0)
            out.append('')
    return '\n'.join(out).rstrip()
