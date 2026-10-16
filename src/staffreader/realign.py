"""Pair a volume's systems with its songs by what they say, not by how
many bars they hold. A trained model reads every staff of every system in
the volume once. Each system is then placed at the song and bar whose
notes it shares most, staff by staff, and kept only where the model's
reading agrees with the transcription there as well as it reads any real
page. The transcriber's line breaks are never needed, so a song paired
with the wrong pages of a large volume, or whose breaks differ from the
print, can still pair system by system."""
import collections
import json
import pathlib

import cv2

from . import canonical, layout, staffcrop, tokens
from .real import LAYOUT_VERSION, SCANS, _found, _parts, _root, answer, pages

MATCH = 0.75  # notes shared, bar by bar, for a system to be placed at all
KEEP = 0.6  # the model's note F1 against the transcription, as the check asks
# Following a single work system by system (_follow): a system is placed
# where its reading agrees best near where the last one ended, by at least
# FOLLOW_MARGIN over the next best start, and kept at FOLLOW_KEEP, since the
# order of the music, not the reading alone, says where it stands.
FOLLOW_KEEP = 0.8  # checked by eye: 0.41 a bar off and 0.60 three bars off (Liszt, Eroica pp. 50, 13)
FOLLOW_MARGIN = 0.1
SHORTEST = 2  # bars; a one-bar system says too little to place
TIE = 0.02  # shares this close are as good as equal


def _notes(bar: tokens.Bar) -> collections.Counter:
    """What a bar shows, as pitches on the staff and rests, in any order."""
    return collections.Counter(n.get('pitch') or 'R' for n in bar.notes if not n.get('space'))


def _shared(a: collections.Counter, b: collections.Counter) -> float:
    if not a and not b:
        return 1.0
    return 2 * sum((a & b).values()) / (sum(a.values()) + sum(b.values()))


def _page_crops(args) -> list[tuple[str, int, object]]:
    """Every staff of every system on a page, cut as the model reads it."""
    from . import model as M
    j, png = args
    found = _found(png)
    if not found['systems']:
        return []
    gray = layout.level(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE), found['angle'], found.get('bend'))
    out = []
    for k, system in enumerate(found['systems']):
        for n, st in enumerate(system['staves']):
            s = found['staves'][st]
            out.append((f'{j + 1}:{k}', n, M.window(staffcrop.crop(gray, s['lines'], s['x0'], s['x1']))))
    return out


def readings(pdf: pathlib.Path, net, name: str, pool=None, batch: int = 128) -> dict[str, list[list[str]]]:
    """The model's reading of every staff of every system in a scan, as
    {'page:system': [tokens of each staff]}, made once and kept beside the
    scan's pages. [pool], if given, cuts the pages in parallel."""
    import torch
    from . import model as M
    pngs = pages(pdf)
    cache = pngs[0].parent / (f'readings-{name}.json' if LAYOUT_VERSION == 1 else f'readings-{name}-layout{LAYOUT_VERSION}.json')
    if cache.exists():
        return json.loads(cache.read_text())
    # A few pages at a time, so a 200-page orchestral score never holds
    # all its staves in memory at once.
    out: dict[str, dict] = {}
    jobs = list(enumerate(pngs))
    for c in range(0, len(jobs), 8):
        chunk = jobs[c:c + 8]
        cut = pool.map(_page_crops, chunk) if pool else map(_page_crops, chunk)
        where, crops = [], []
        for page in cut:
            for key, n, crop in page:
                where.append((key, n))
                crops.append(crop)
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1])
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            x, w = M.batch_images([crops[i] for i in idx], 'cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                got = net.read(x, w)
            for i, seq in zip(idx, got):
                key, n = where[i]
                out.setdefault(key, {})[n] = seq
    out = {k: [v[n] for n in sorted(v)] for k, v in out.items()}
    cache.write_text(json.dumps(out))
    return out


class Song:
    """A transcription cut staff by staff into bars, each with what it
    shows, so a system can be placed in it."""

    def __init__(self, mxl: pathlib.Path):
        self.mxl = mxl
        root = _root(canonical.load(mxl))
        self.parts = list(_parts(root))
        self.count = len(self.parts[0][0].findall('measure'))
        self.cut = {(k, st): tokens.from_part(part, st)
                    for k, (part, c) in enumerate(self.parts) for st in range(1, c + 1)}
        self.places = sorted(self.cut)
        self.bars = {}
        for place, (seq, states) in self.cut.items():
            bars = tokens.parse(tokens.stretch(seq, states, 0, self.count - 1).tokens)
            notes = [_notes(b) for b in bars[:self.count]]
            self.bars[place] = notes + [collections.Counter()] * (self.count - len(notes))

    def together(self, a, b, first: int, last: int) -> list[str] | None:
        """Two single-staff parts printed on one staff (tokens.merged_part)."""
        if (a, b) not in self.cut:
            both = tokens.merged_part(self.parts[a[0]][0], self.parts[b[0]][0])
            if both is None:
                return None
            self.cut[(a, b)] = tokens.from_part(both, 1)
        return self.tokens((a, b), first, last)

    def tokens(self, place, first: int, last: int) -> list[str] | None:
        seq, states = self.cut[place]
        if not all(x['readable'] for x in states[first:last + 1]):
            return None
        return tokens.stretch(seq, states, first, last).tokens


def place(read: list[list[collections.Counter]], songs: list[Song],
          after: tuple[int, int] | None = None) -> tuple[float, int, int] | None:
    """The song and first bar where a system's bars, as read staff by
    staff, share the most notes with the transcription: (share, song, bar).
    A song's verses often repeat the same pitches, so among places that
    share about as much, the one that goes on from [after] (the song and
    last bar of the system placed before) wins, then the next one later in
    that song, then the first."""
    bars = len(read[0])
    found = []
    for s, song in enumerate(songs):
        if len(song.places) != len(read) or song.count < bars:
            continue
        for first in range(song.count - bars + 1):
            total = 0.0
            for n, p in enumerate(song.places):
                want = song.bars[p]
                total += sum(_shared(read[n][t], want[first + t]) for t in range(bars))
            found.append((total / (bars * len(read)), s, first))
    if not found:
        return None
    best = max(f[0] for f in found)
    near = [f for f in found if f[0] >= best - TIE]
    if after is not None:
        song, last = after
        if goes_on := [f for f in near if f[1] == song and f[2] == last + 1]:
            return goes_on[0]
        if later := sorted((f for f in near if f[1] == song and f[2] > last), key=lambda f: f[2]):
            return later[0]
    return max(near, key=lambda f: f[0])


def _in_order(plan: list[list]) -> int:
    """Keep only placements that follow one another: in each song, the
    longest run of systems whose first bars rise in page order, and only
    where it has more than one system. A volume holds songs that were never
    transcribed, and a system of one can resemble a transcribed song closely
    enough to be placed in it; a test set, which keeps every placement, must
    not keep those. Returns how many placements it dropped."""
    by_song: dict[int, list[int]] = {}
    for i, p in enumerate(plan):
        if p[2] is not None:
            by_song.setdefault(p[2][1], []).append(i)
    dropped = 0
    for s, idx in by_song.items():
        firsts = [plan[i][2][2] for i in idx]
        # The longest strictly rising run of first bars, in page order.
        best, back = [1] * len(idx), [-1] * len(idx)
        for b in range(len(idx)):
            for a in range(b):
                if firsts[a] < firsts[b] and best[a] + 1 > best[b]:
                    best[b], back[b] = best[a] + 1, a
        end = max(range(len(idx)), key=lambda k: best[k])
        keep = set()
        while end != -1:
            keep.add(idx[end])
            end = back[end]
        if len(keep) < 2:
            keep = set()
        for i in idx:
            if i not in keep:
                plan[i][2] = None
                dropped += 1
    return dropped


def _fill(plan: list[list], songs: list) -> None:
    """Place each system left unplaced between two placed systems of the
    same song whose gap its bars fill exactly: a test set keeps a system a
    model read too badly to place, where the music's order says where it
    stands. [plan] holds [key, bars, placement or None] in page order."""
    placed = [i for i, p in enumerate(plan) if p[2] is not None]
    for a, b in zip(placed, placed[1:]):
        (sa, fa), (sb, fb) = plan[a][2][1:], plan[b][2][1:]
        if sa != sb or b - a < 2:
            continue
        gap = list(range(a + 1, b))
        start = fa + plan[a][1]
        if start + sum(plan[i][1] for i in gap) != fb:
            continue
        for i in gap:
            plan[i][2] = (0.0, sa, start)
            start += plan[i][1]


def _shown(song, first: int, count: int, n: int) -> list | None:
    """The song's staves a system of [n] staves prints for bars
    first..first+count-1: all of them, or where the transcription has more
    (a third piano staff for a few passages), those that hold notes there."""
    if len(song.places) == n:
        return list(song.places)
    held = [p for p in song.places
            if any(t.startswith('N:') for t in (song.tokens(p, first, first + count - 1) or []))]
    return held if len(held) == n else None


def _agree(song, staves: list[list[str]], first: int, count: int, places=None) -> float | None:
    """The mean note F1 of a system's readings against its song's bars
    first..first+count-1, staff by staff."""
    from . import measure, metrics
    f1 = []
    for p, seq in zip(places or song.places, staves):
        want = song.tokens(p, first, first + count - 1)
        if want is None:
            continue
        try:
            f1.append(metrics.compare(measure._score(want), measure._score(seq)).notes['core'].f1)
        except Exception:
            f1.append(0.0)
    return sum(f1) / len(f1) if f1 else None


def _follow(plan: list[list], seen: dict, song) -> int:
    """Place the systems of a scan that holds one work by following the
    music in page order: each unplaced system is tried at starts near where
    the system before it ended, and placed at the start its reading agrees
    with best, where that start is clearly better than the next. A system
    the model reads too badly to place by its exact bars (dense piano
    writing) is placed this way. Returns how many it placed."""
    pointer, placed = 0, 0
    for row in plan:
        key, count, got = row
        if got is not None:
            pointer = got[2] + count
            continue
        staves = seen[key]
        if count < 1 or pointer >= song.count:
            continue
        tried = []
        for first in range(max(0, pointer - 2), min(song.count - count, pointer + 4) + 1):
            places = _shown(song, first, count, len(staves))
            f1 = _agree(song, staves, first, count, places) if places else None
            if f1 is not None:
                # Going on where the system before ended is itself evidence.
                tried.append((f1 + (FOLLOW_MARGIN if first == pointer else 0.0), f1, first, places))
        tried.sort(key=lambda t: -t[0])
        if not tried:
            pointer += count  # one work in order: the next system starts after this one
            continue
        score, best, first, places = tried[0]
        runner = tried[1][0] if len(tried) > 1 else 0.0
        if best >= FOLLOW_KEEP and score - runner >= FOLLOW_MARGIN:
            row[2] = (best, 0, first, 'follow', places)
            placed += 1
            pointer = first + count
        else:
            pointer += count
    return placed


def _stem(song) -> str:
    """How a pair names its transcription: openscore:<stem> for OpenScore's,
    else the file's path."""
    from .real import ANSWERS
    from .render import CACHE
    orchestral = CACHE / 'openscore' / 'orchestra' / 'answers'
    return f'openscore:{song.mxl.stem}' if song.mxl.parent in (ANSWERS, orchestral) else str(song.mxl)


def pair_volume(pdf: pathlib.Path, rows: list[dict], net, model_name: str, out: pathlib.Path,
                name: str, keep: float = KEEP, fill: bool = False, follow: bool = False) -> tuple[list[dict], dict]:
    """Pair every system of [pdf] that can be placed in one of its songs
    ([rows], as pair_all takes them), and cut its staves. A system is kept
    where the model's reading agrees with the transcription at [keep];
    [fill] also places systems by the music's order (_fill), as a test set
    does with keep 0."""
    from . import measure, metrics
    songs, ids = [], []
    for r in rows:
        try:
            songs.append(Song(pathlib.Path(r['mxl']) if r.get('mxl')
                              else answer(r['path'], r['id'], r.get('collection', 'lieder'))))
            ids.append(r['id'])
        except Exception:
            continue
    seen = readings(pdf, net, model_name)
    pngs = pages(pdf)
    report = {'pdf': pdf.name, 'songs': len(songs), 'systems': len(seen), 'placed': 0, 'kept': 0}
    plan, after = [], None
    for key, staves in sorted(seen.items(), key=lambda kv: tuple(int(x) for x in kv[0].split(':'))):
        j, k = (int(x) for x in key.split(':'))
        count = _found(pngs[j - 1])['systems'][k]['bars']
        read = [tokens.parse(seq) for seq in staves]
        got = None
        if count >= SHORTEST and all(len(b) == count for b in read):
            got = place([[_notes(b) for b in bars] for bars in read], songs, after)
            if got is not None and got[0] < MATCH:
                got = None
        if got is not None:
            after = (got[1], got[2] + count - 1)
        plan.append([key, count, got])
    if fill:
        report['out of order'] = _in_order(plan)
        _fill(plan, songs)
    if follow and len(songs) == 1:
        report['followed'] = _follow(plan, seen, songs[0])
    samples, grays = [], {}
    for key, count, got in plan:
        if got is None:
            continue
        j, k = (int(x) for x in key.split(':'))
        found = _found(pngs[j - 1])
        staves = seen[key]
        report['placed'] += 1
        share, s, first = got[:3]
        followed = len(got) > 3
        song, last = songs[s], first + count - 1
        places = got[4] if followed else song.places
        if len(places) != len(staves):
            continue
        want = [song.tokens(p, first, last) for p in places]
        f1 = []
        for w, seq in zip(want, staves):
            if w is None:
                continue
            try:
                f1.append(metrics.compare(measure._score(w), measure._score(seq)).notes['core'].f1)
            except Exception:
                f1.append(0.0)
        if not f1 or sum(f1) / len(f1) < (FOLLOW_KEEP if followed else keep):
            continue
        report['kept'] += 1
        if j not in grays:
            grays = {j: layout.level(cv2.imread(str(pngs[j - 1]), cv2.IMREAD_GRAYSCALE), found['angle'], found.get('bend'))}
        system = found['systems'][k]
        for n, ((part, staff_no), w) in enumerate(zip(places, want)):
            if w is None:
                continue
            st = found['staves'][system['staves'][n]]
            file = out / name / f'p{j:04d}-s{k}-{part}-{staff_no}.png'
            file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(file), staffcrop.crop(grays[j], st['lines'], st['x0'], st['x1']))
            samples.append({'image': str(file.relative_to(out)), 'mxl': _stem(song),
                            'part': part, 'staff': staff_no, 'first': first, 'last': last, 'font': 'scan',
                            'pdf': pdf.name, 'page': j, 'at': system['staves'][n], 'tokens': w,
                            'match': round(share, 3), **({'followed': True} if followed else {})})
    return samples, report


def _pair_one(args):
    pdf, rows, model_name, out, by_staff, follow = args
    try:
        if by_staff:
            return pair_staves(pdf, rows, None, model_name, out, pdf.stem.split('-')[0])
        return pair_volume(pdf, rows, None, model_name, out, pdf.stem.split('-')[0], follow=follow)
    except Exception as e:
        return [], {'pdf': pdf.name, 'error': repr(e)[:300]}


def pair_all(rows: list[dict], name: str, model: pathlib.Path, workers: int = 8, by_staff: bool = False,
             follow: bool = False) -> pathlib.Path:
    """Pair every volume on disk with the songs it holds, by content: the
    model reads every volume first, then the systems are placed in
    parallel."""
    from concurrent.futures import ProcessPoolExecutor
    from . import model as M
    from .progress import Progress
    from .render import CACHE
    out = CACHE / 'data' / name
    out.mkdir(parents=True, exist_ok=True)
    by_pdf = collections.defaultdict(list)
    for r in rows:
        # A row may name its scan outright: a candidate copy, not yet IMSLP's own file.
        pdf = pathlib.Path(r['pdf']) if r.get('pdf') else next(SCANS.glob(f"IMSLP{r['imslp']}-*.pdf"), None)
        if pdf is not None:
            by_pdf[pdf].append(r)
    volumes = sorted(by_pdf, key=lambda p: p.name)
    model_name = f'{model.parent.name}-{model.stem}'
    job = Progress(f'realign-{name}', 2, 'stages', 'placing each scanned system in its song by what it says')
    net = M.load(model)
    bar = Progress(f'realign-{name} read', len(volumes), 'volumes', f'reading every staff with {model_name}')
    with ProcessPoolExecutor(workers) as pool:
        for v, pdf in enumerate(volumes, start=1):
            try:
                readings(pdf, net, model_name, pool)
            except Exception as e:
                print(json.dumps({'pdf': pdf.name, 'error': repr(e)[:300]}), flush=True)
            bar.update(v, pdf.name[:50])
    bar.finish(f'{len(volumes)} volumes read')
    job.update(1, 'read; placing')
    del net
    bar = Progress(f'realign-{name} place', len(volumes), 'volumes', 'placing systems in songs')
    reports = []
    with open(out / 'index.jsonl', 'w') as index, ProcessPoolExecutor(workers) as pool:
        # The largest volumes first, so the last to finish are small.
        work = sorted(volumes, key=lambda p: -len(by_pdf[p]))
        for v, (samples, report) in enumerate(pool.map(_pair_one, [(p, by_pdf[p], model_name, out, by_staff, follow) for p in work]),
                                              start=1):
            reports.append(report)
            for s in samples:
                index.write(json.dumps(s) + '\n')
            index.flush()
            print(json.dumps(report), flush=True)
            bar.update(v, f'{sum(r.get("kept", 0) for r in reports)} systems kept so far')
    (out / 'report.json').write_text(json.dumps(reports, indent=1))
    done = f'{sum(r.get("kept", 0) for r in reports)} kept ({"staves" if by_staff else "systems"}), of {sum(r.get("systems", 0) for r in reports)} systems'
    bar.finish(done)
    job.finish(done)
    return out


# Orchestral scores: a printed staff often holds two players (two flutes
# on one staff), and empty staves are left out of a system, so a system
# matches no fixed set of the transcription's parts. Each staff is
# matched to a part on its own.
KEEP_STAFF = 0.95  # a staff holding two players matches either one's part only in part; checked by eye at 0.857, partly wrong


def _key(c: collections.Counter) -> frozenset | None:
    """A bar's notes as a lookup key; a bar of rests or a single note
    says too little to place."""
    if sum(n for p, n in c.items() if p != 'R') < 2:
        return None
    return frozenset(c.items())


def pair_staves(pdf: pathlib.Path, rows: list[dict], net, model_name: str, out: pathlib.Path,
                name: str, keep: float = KEEP_STAFF, fill: bool = False) -> tuple[list[dict], dict]:
    """Pair every staff of [pdf] that can be matched to one part of one
    of its movements ([rows]). A system is placed by its staves' votes:
    each read bar that equals a bar of some part votes for the movement
    and first bar that puts it there. At that place each staff takes the
    part it agrees with best, each part once, and is kept where the model's
    reading of it agrees with that part's transcription at [keep]. [fill]
    also places systems by the music's order (_fill), as a test set does
    with keep 0."""
    from . import measure, metrics
    songs = []
    for r in rows:
        try:
            songs.append(Song(pathlib.Path(r['mxl']) if r.get('mxl')
                              else answer(r['path'], r['id'], r.get('collection', 'lieder'))))
        except Exception:
            continue
    index = collections.defaultdict(list)
    for s, song in enumerate(songs):
        for place in song.places:
            for i, c in enumerate(song.bars[place]):
                if (k := _key(c)) is not None:
                    index[k].append((s, i))
    seen = readings(pdf, net, model_name)
    pngs = pages(pdf)
    report = {'pdf': pdf.name, 'songs': len(songs), 'systems': len(seen), 'placed': 0, 'staves': 0, 'kept': 0}
    plan, after = [], None
    for key, staves in sorted(seen.items(), key=lambda kv: tuple(int(x) for x in kv[0].split(':'))):
        j, k = (int(x) for x in key.split(':'))
        count = _found(pngs[j - 1])['systems'][k]['bars']
        read = [[_notes(b) for b in tokens.parse(seq)] for seq in staves]
        plan.append([key, count, None])
        if count < SHORTEST:
            continue
        votes = collections.Counter()
        for bars in read:
            if len(bars) != count:
                continue
            mine = set()
            for t, c in enumerate(bars):
                for s, i in index.get(_key(c), ()) if _key(c) is not None else ():
                    if 0 <= i - t <= songs[s].count - count:
                        mine.add((s, i - t))
            votes.update(mine)
        if not votes:
            continue
        (s, first), best = votes.most_common(1)[0]
        if after and votes.get((after[0], after[1] + 1), 0) >= 0.8 * best:
            s, first = after[0], after[1] + 1
        if votes[(s, first)] < 2:
            continue
        after = (s, first + count - 1)
        plan[-1][2] = (float(best), s, first)
    if fill:
        report['out of order'] = _in_order(plan)
        _fill(plan, songs)
    samples, grays = [], {}
    for key, count, got in plan:
        if got is None:
            continue
        _, s, first = got
        j, k = (int(x) for x in key.split(':'))
        found = _found(pngs[j - 1])
        staves = seen[key]
        read = [[_notes(b) for b in tokens.parse(seq)] for seq in staves]
        song, last = songs[s], first + count - 1
        report['placed'] += 1
        # Each staff takes the part it shares most notes with, best first.
        pairs = sorted(((sum(_shared(read[n][t], song.bars[p][first + t]) for t in range(count)) / count, n, p)
                        for n in range(len(read)) if len(read[n]) == count for p in song.places), reverse=True)
        used_n, used_p, chosen = set(), set(), []
        for share, n, p in pairs:
            if n in used_n or p in used_p or share < 0.5:
                continue
            used_n.add(n)
            used_p.add(p)
            chosen.append((n, p))

        def agree(want, n):
            try:
                return metrics.compare(measure._score(want), measure._score(staves[n])).notes['core'].f1
            except Exception:
                return None

        for n, p in chosen:
            report['staves'] += 1
            want = song.tokens(p, first, last)
            if want is None:
                continue
            f1 = agree(want, n)
            if f1 is None:
                continue
            # Two players printed on one staff (two flutes): a neighboring
            # part this system gives no staff of its own, whose notes this
            # staff also holds, and which fits it clearly better together.
            partner = None
            for q in song.places:
                if q in used_p or p[1] != 1 or q[1] != 1 or abs(q[0] - p[0]) != 1 \
                        or self_count(song, q) != 1 or self_count(song, p) != 1:
                    continue
                if sum(_shared(read[n][t], song.bars[q][first + t]) for t in range(count)) / count < 0.5:
                    continue
                a, b = (p, q) if p[0] < q[0] else (q, p)
                both = song.together(a, b, first, last)
                f1_both = agree(both, n) if both is not None else None
                if f1_both is not None and f1_both >= f1 + 0.1:
                    want, f1, partner = both, f1_both, q
                    used_p.add(q)
                    report['shared'] = report.get('shared', 0) + 1
                    break
            if f1 < keep:
                continue
            report['kept'] += 1
            if j not in grays:
                grays = {j: layout.level(cv2.imread(str(pngs[j - 1]), cv2.IMREAD_GRAYSCALE), found['angle'],
                                         found.get('bend'))}
            st = found['staves'][found['systems'][k]['staves'][n]]
            part, staff_no = p
            file = out / name / f'p{j:04d}-s{k}-n{n}.png'
            file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(file), staffcrop.crop(grays[j], st['lines'], st['x0'], st['x1']))
            if partner is not None:
                part = min(p[0], partner[0])
            samples.append({'image': str(file.relative_to(out)), 'mxl': _stem(song),
                            'part': part, 'staff': staff_no, 'first': first, 'last': last, 'font': 'scan',
                            'pdf': pdf.name, 'page': j, 'tokens': want, 'check_f1': round(f1, 3),
                            **({'with': max(p[0], partner[0])} if partner is not None else {})})
    return samples, report


def self_count(song: Song, place) -> int:
    """How many staves the part of [place] has."""
    return song.parts[place[0]][1]
