"""Real staves to train on: pages of the editions OpenScore transcribed,
each staff paired with the exact tokens it prints.

OpenScore's transcriptions keep their source's page and system breaks.
So once the layout stage has found a page's systems and the bars in
each, the page's systems line up with the transcription's: the same
number of systems, each with the same number of bars. A page that agrees
exactly gives every staff of every system; a page with the right number
of systems gives each system whose bar count agrees; a page that does
not line up gives nothing. The scan may hold more than the work, such as
a title page or the end of the work before it, so the transcription's
first page is placed where the most pages agree."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time
import xml.etree.ElementTree as ET

import cv2
import requests

from . import canonical, layout, staffcrop, tokens
from .data import _parts, _root, _systems
from . import contact
from .render import CACHE
from . import settings

OPENSCORE = 'https://raw.githubusercontent.com/OpenScore/StringQuartets'
COMMIT = '9be3df2ace482130fe031b9e8a647cdf112ed243'
# Each OpenScore collection, at a pinned commit, with its files' prefix.
COLLECTIONS = {
    'quartets': ('https://raw.githubusercontent.com/OpenScore/StringQuartets', COMMIT, 'sq'),
    'lieder': ('https://raw.githubusercontent.com/OpenScore/Lieder', '38c5db510224d9facdc4b08d741fc788cfb58ea8', 'lc'),
}
ANSWERS = CACHE / 'openscore' / 'answers'
PAGES = CACHE / 'real' / 'pages'
SCANS = settings.path('scans')


def answer(path: str, score_id: str, collection: str = 'quartets') -> pathlib.Path:
    """OpenScore's transcription of a quartet, at the pinned commit: its
    MusicXML, and beside it the MuseScore file, which alone tells the
    transcriber's own line breaks from those MuseScore added."""
    ANSWERS.mkdir(parents=True, exist_ok=True)
    repo, commit, prefix = COLLECTIONS[collection]
    out = ANSWERS / f'{prefix}{score_id}.mxl'
    base = f'{repo}/{commit}/scores/{path}/{prefix}{score_id}'
    if not out.exists():
        got = _get(f'{base}.mxl')
        if got is None:
            # Only the MuseScore file: MuseScore exports its own file, the
            # one conversion trusted to keep everything the file holds.
            packed = _get(f'{base}.mscz', required=True)
            source = out.with_suffix('.export.mscz')
            source.write_bytes(packed)
            from .render import MSCORE, mscore_env
            subprocess.run([MSCORE, '-o', str(out.with_suffix('.musicxml')), str(source)], env=mscore_env(),
                           capture_output=True, timeout=300)
            got = out.with_suffix('.musicxml').read_bytes()
        out.write_bytes(got)
    mscx = out.with_suffix('.mscx')
    if not mscx.exists() and not out.with_suffix('.none').exists():
        got = _get(f'{base}.mscx')
        if got is None and (packed := _get(f'{base}.mscz')) is not None:
            import io
            import zipfile
            with zipfile.ZipFile(io.BytesIO(packed)) as z:
                inner = next((n for n in z.namelist() if n.endswith('.mscx')), None)
                got = z.read(inner) if inner else None
        if got is None:
            out.with_suffix('.none').touch()  # this transcription has no MuseScore file to read
        else:
            mscx.write_bytes(got)
    return out


def _get(url: str, required: bool = False) -> bytes | None:
    time.sleep(2)
    r = requests.get(url, headers={'User-Agent': contact.user_agent()}, timeout=120)
    if r.status_code == 404 and not required:
        return None
    r.raise_for_status()
    return r.content


def written_breaks(mscx: pathlib.Path, count: int) -> list[int] | None:
    """The first measure of each system as the transcriber broke the lines,
    or None if the MuseScore file's measures do not match the MusicXML's."""
    if not mscx.exists():
        return None
    staff = ET.parse(mscx).getroot().find('.//Score/Staff')
    if staff is None:
        return None
    measures = staff.findall('Measure')
    if len(measures) != count:
        return None
    starts = [0]
    for i, m in enumerate(measures[:-1]):
        if any(b.findtext('subtype') in ('line', 'page', 'section') for b in m.iter('LayoutBreak')):
            starts.append(i + 1)
    return starts


MAX_SIDE = 6000  # pixels, the longest side of a page drawn from a scan


def pages(pdf: pathlib.Path, dpi: int = 300) -> list[pathlib.Path]:
    """Every page of a scan as a gray PNG, made once, under a lock so two
    works sharing one scan never read a page half written."""
    import fcntl
    out = PAGES / pdf.stem[:80]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (out / '.done').exists():
            if pdf.is_dir():  # page pictures, as Commons or MDZ give them
                for k, picture in enumerate(sorted(pdf.glob('*.jpg')), start=1):
                    cv2.imwrite(str(out / f'page-{k:04d}.png'), cv2.imread(str(picture), cv2.IMREAD_GRAYSCALE))
            else:
                # A scan whose pages are declared very large is drawn smaller,
                # its longest side at most MAX_SIDE pixels: at 300 dpi one such
                # page took gigabytes once flattened.
                info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
                size = next((l.split(':', 1)[1].split('pts')[0].split('x') for l in info.splitlines()
                             if l.startswith('Page size:')), None)
                if size:
                    longest = max(float(v) for v in size) / 72
                    dpi = min(dpi, max(72, int(MAX_SIDE / longest)))
                subprocess.run(['pdftoppm', '-r', str(dpi), '-gray', '-png', str(pdf), str(out / 'page')],
                               check=True)
            (out / '.done').touch()
    return sorted(out.glob('page-*.png'))


# Each version of the layout reader caches its own page layouts, so a set
# paired with an older one can be rebuilt as it was. 2: pages flattened
# where their paper bends. 3: the tilt measured across the music's width,
# and a line a long beam thickens kept. 4: a system joined by the line at
# its left edge where its staves run on into brackets and braces.
LAYOUT_VERSION = 4


def _found(png: pathlib.Path) -> dict:
    """A page's layout, cached beside it."""
    cached = png.with_suffix(f'.layout{LAYOUT_VERSION}.json' if LAYOUT_VERSION > 1 else '.layout.json')
    if cached.exists():
        return json.loads(cached.read_text())
    lay = layout.read(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE))
    out = {'angle': lay.angle, 'bend': lay.bend,
           'staves': [{'lines': s.lines, 'x0': s.x0, 'x1': s.x1} for s in lay.staves],
           'systems': [{'staves': sy.staves, 'bars': sy.measures} for sy in lay.systems]}
    cached.write_text(json.dumps(out))
    return out


def _bars(found: dict, staves: int) -> list[int] | None:
    if any(len(s['staves']) != staves for s in found['systems']):
        return None
    return [s['bars'] for s in found['systems']]


def align(want: list[int], seen: list[int | None], gap: float = 0.2, run: int = 3) -> list[tuple[int, int]]:
    """Line up the transcription's systems (their bar counts, in order
    through the work) with the systems found on the scan (None where a
    system is not whole), letting either side skip a system. A pair is
    kept only inside a run of [run] systems that agree one after another
    on both sides, so a lone coincidence never pairs."""
    n, m = len(want), len(seen)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0], move[i][0] = -gap * i, 1
    for j in range(1, m + 1):
        score[0][j], move[0][j] = -gap * j, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, how = score[i - 1][j] - gap, 1
            if score[i][j - 1] - gap > best:
                best, how = score[i][j - 1] - gap, 2
            if seen[j - 1] is not None and seen[j - 1] == want[i - 1] and score[i - 1][j - 1] + 1 > best:
                best, how = score[i - 1][j - 1] + 1, 3
            score[i][j], move[i][j] = best, how
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        how = move[i][j]
        if how == 3:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif how == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    keep = []
    k = 0
    while k < len(pairs):
        e = k
        while e + 1 < len(pairs) and pairs[e + 1] == (pairs[e][0] + 1, pairs[e][1] + 1):
            e += 1
        if e - k + 1 >= run:
            keep += pairs[k:e + 1]
        k = e + 1
    return keep


def pair(pdf: pathlib.Path, mxl: pathlib.Path, out: pathlib.Path, name: str) -> tuple[list[dict], dict]:
    """Pair a scan's systems with its transcription's, and cut the staves
    of every system that pairs. Returns the samples and what happened."""
    root = _root(canonical.load(mxl))
    parts = list(_parts(root))
    staves = sum(c for _, c in parts)
    first_staff = [sum(c for _, c in parts[:k]) for k in range(len(parts))]
    count = len(parts[0][0].findall('measure'))
    pngs = pages(pdf)
    found = [_found(p) for p in pngs]
    where = [(j, k) for j, f in enumerate(found) for k in range(len(f['systems']))]
    seen = [found[j]['systems'][k]['bars'] if len(found[j]['systems'][k]['staves']) == staves else None
            for j, k in where]
    # The transcriber's own breaks, or MuseScore's layout of the score,
    # whichever lines up with the scan better.
    best = None
    for kind, starts in (('written', written_breaks(mxl.with_suffix('.mscx'), count)),
                         ('exported', [m for page in _systems(root) for m in page])):
        if not starts:
            continue
        bars = [b - a for a, b in zip(starts, starts[1:] + [count])]
        got = align(bars, seen)
        if best is None or len(got) > len(best[3]):
            best = (kind, starts, bars, got)
    kind, starts, bars, pairs = best
    report = {'pdf': pdf.name, 'pages': len(pngs), 'breaks': kind, 'systems': len(bars), 'found': len(seen),
              'found_whole': sum(x is not None for x in seen), 'systems_kept': len(pairs)}
    # Every staff of every part: a voice, and a piano's two staves.
    cut = {(k, st): tokens.from_part(part, st) for k, (part, c) in enumerate(parts) for st in range(1, c + 1)}
    samples, grays = [], {}
    for i, jj in pairs:
        j, s_i = where[jj]
        first, last = starts[i], starts[i] + bars[i] - 1
        if j not in grays:
            grays = {j: layout.level(cv2.imread(str(pngs[j]), cv2.IMREAD_GRAYSCALE), found[j]['angle'], found[j].get('bend'))}
        system = found[j]['systems'][s_i]
        for (part, staff_no), (seq, states) in cut.items():
            if not all(x['readable'] for x in states[first:last + 1]):
                continue
            st = found[j]['staves'][system['staves'][first_staff[part] + staff_no - 1]]
            image = staffcrop.crop(grays[j], st['lines'], st['x0'], st['x1'])
            file = out / name / f'p{j + 1:03d}-s{s_i}-{part}-{staff_no}.png'
            file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(file), image)
            samples.append({'image': str(file.relative_to(out)), 'mxl': f'openscore:{mxl.stem}', 'part': part,
                            'staff': staff_no,
                            'first': first, 'last': last, 'font': 'scan', 'pdf': pdf.name, 'page': j + 1,
                            'tokens': tokens.stretch(seq, states, first, last).tokens})
    return samples, report


def _pair_one(args):
    r, out = args
    pdf = next(SCANS.glob(f"IMSLP{r['imslp']}-*.pdf"), None)
    if pdf is None:
        return [], None
    collection = r.get('collection', 'quartets')
    try:
        mxl = answer(r['path'], r['id'], collection)
        samples, report = pair(pdf, mxl, out, f"{COLLECTIONS[collection][2]}{r['id']}")
    except Exception as e:
        return [], {'pdf': pdf.name, 'work': r['path'], 'error': repr(e)[:300]}
    report['work'] = r['path']
    return samples, report


def pair_all(rows: list[dict], name: str = 'real-a', workers: int = 8) -> pathlib.Path:
    """Pair every scan on disk that is the source of an OpenScore quartet."""
    from concurrent.futures import ProcessPoolExecutor
    out = CACHE / 'data' / name
    out.mkdir(parents=True, exist_ok=True)
    from .progress import Progress
    reports = []
    bar = Progress(f'pair-{name}', len(rows), 'works', 'pairing scans with OpenScore transcriptions')
    with open(out / 'index.jsonl', 'w') as index, ProcessPoolExecutor(workers) as pool:
        for k, (samples, report) in enumerate(pool.map(_pair_one, [(r, out) for r in rows]), start=1):
            bar.update(k, f'{sum(r.get("systems_kept", 0) for r in reports)} systems kept so far')
            if report is None:
                continue
            reports.append(report)
            for s in samples:
                index.write(json.dumps(s) + '\n')
            index.flush()
            print(json.dumps(report), flush=True)
    (out / 'report.json').write_text(json.dumps(reports, indent=1))
    bar.finish(f'{sum(r.get("systems_kept", 0) for r in reports)} systems kept')
    return out
