"""Running readers over the ladder and scoring what they read.

Each reader reads each page of each track of each case; the pages'
readings are joined part by part and compared with the case's bars. A
reading is kept in the cache, keyed by the picture it was read from, so a
run repeats only what changed. A reader that fails on a page scores as
having read nothing there."""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
import traceback
from dataclasses import dataclass

from . import canonical, metrics, render
from .cases import Case, Track


@dataclass
class Result:
    case: str
    level: int
    track: str
    reader: str
    summary: dict
    seconds: float
    error: str | None
    readings: list[str]
    bars: list[dict]


def _digest(path: pathlib.Path, page: int | None) -> str:
    h = hashlib.sha256(path.read_bytes())
    h.update(str(page).encode())
    return h.hexdigest()[:16]


def join(scores: list[canonical.Score]) -> canonical.Score:
    """Readings of consecutive pages as one score: each page's parts are
    appended to the parts in the same position."""
    parts: list[canonical.Part] = []
    for s in scores:
        for i, p in enumerate(s.parts):
            if i < len(parts):
                # A page opens with its clefs and key, as every system does;
                # they change only where they differ from the page before.
                if p.measures and parts[i].measures:
                    before, first = parts[i].measures[-1], p.measures[0]
                    first.changes = frozenset(
                        c for c in first.changes
                        if (c == 'key' and first.key != before.key) or (c == 'time' and first.time != before.time)
                        or (c.startswith('clef') and first.clefs.get(int(c[4:])) != before.clefs.get(int(c[4:]))))
                parts[i].measures += p.measures
            else:
                parts.append(canonical.Part(p.id, p.name, p.staves, list(p.measures), p.transpose))
    for p in parts:
        for i, m in enumerate(p.measures):
            m.index = i
    return canonical.Score(parts)


def run_one(case: Case, track: Track, reader, fresh: bool = False) -> Result:
    readings, scores, error, seconds = [], [], None, 0.0
    for image, page in case.pages(track):
        work = render.CACHE / 'readings' / reader.NAME / _digest(image, page)
        out = work / 'reading.json'
        failed = work / 'error.txt'
        if fresh:
            failed.unlink(missing_ok=True)
        if out.exists() and not fresh:
            cached = json.loads(out.read_text())
            seconds += cached.get('seconds', 0.0)
            readings.append(cached['musicxml'])
            scores.append(canonical.read(cached['musicxml']))
            continue
        if failed.exists():  # a reader that failed on this picture fails again; --fresh retries it
            error = failed.read_text().splitlines()[0]
            continue
        started = time.time()
        try:
            xml = reader.read(image, page, work)
            took = round(time.time() - started, 1)
            seconds += took
            out.write_text(json.dumps({'musicxml': str(xml), 'image': str(image), 'page': page, 'seconds': took}))
            readings.append(str(xml))
            scores.append(canonical.read(xml))
        except Exception as e:  # a reader that fails reads nothing
            seconds += round(time.time() - started, 1)
            error = f'{type(e).__name__}: {e}'.strip().splitlines()[0]
            work.mkdir(parents=True, exist_ok=True)
            failed.write_text(f'{error}\n\n{traceback.format_exc()}')
    truth = case.truth(track)
    reading = join(scores) if scores else canonical.Score([])
    report = metrics.compare(truth, reading)
    bars = [{'part': b.part, 'gt': b.gt, 'pred': b.pred, 'exact_core': b.exact_core, 'exact_full': b.exact_full,
             'missed': b.notes_missed, 'extra': b.notes_extra, 'notes': b.notes_gt, 'valid': b.valid,
             'number': truth.parts[b.part].measures[b.gt].number if b.gt is not None else None}
            for b in report.bars]
    return Result(case.id, case.level, track.name, reader.NAME, report.summary(case.masks),
                  round(seconds, 1), error, readings, bars)


def run(cases: list[Case], readers: list, out: pathlib.Path, fresh: bool = False) -> list[Result]:
    from .progress import Progress
    out.mkdir(parents=True, exist_ok=True)
    results = []
    bar = Progress(f'ladder {out.name}', sum(len(c.tracks) for c in cases) * len(readers), 'readings',
                   f"reading the ladder with {', '.join(r.NAME for r in readers)}")
    for case in cases:
        for track in case.tracks:
            for reader in readers:
                bar.update(len(results), f'{case.id} {track.name} {reader.NAME}')
                r = run_one(case, track, reader, fresh)
                s = r.summary
                print(f'{case.id:24} {track.name:18} {reader.NAME:10} core F1 {s["note_f1_core"]:.3f}  '
                      f'bars exact {s["bar_exact_full"]:.3f}  {r.seconds:6.1f}s' + (f'  ERROR {r.error[:80]}' if r.error else ''),
                      flush=True)
                results.append(r)
                (out / 'results.json').write_text(json.dumps([r.__dict__ for r in results], indent=1, default=str))
    bar.finish(str(out / 'results.json'))
    return results
