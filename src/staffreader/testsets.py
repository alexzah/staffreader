"""The reserved test sets (docs/improving.md, step 1): every staff of the scans
the reader never trains on, paired with their transcriptions and kept
however well any model reads them.

The scans are the ladder cases' and those reserved.yaml names
(train.reserved). Each is paired with every transcription made from it:
the ladder case's answer, OpenScore's orchestral movements and Lieder
songs. A system is placed by its content, or by the music's order between
placed systems (realign._fill), and kept whatever its reading scores.
The staves are labeled in format 2, a system's first staff marked as its
top. The sets are written to data/test-<music>.

The verses sung under the reserved voice staves are cut as text lines
for the text reader, in data/test-lyrics."""

from __future__ import annotations

import collections
import json
import pathlib
import re

import cv2
import yaml

from . import settings

from . import cases, marks, realign, tokens, train
from .data import _parts, _root
from . import canonical
from .real import ANSWERS
from .render import CACHE

ORCHESTRA = CACHE / 'openscore' / 'orchestra'
LIEDER = CACHE / 'openscore' / 'lieder-scores-pinned.yaml'


def _number(pdf: str) -> str:
    return pdf.split('-')[0].removeprefix('IMSLP').lstrip('0')


def plan() -> dict[str, dict]:
    """For each reserved scan on disk: its kind of music, how to pair it
    (whole systems, or staff by staff for an orchestral score), and the
    transcriptions made from it, as pairing rows."""
    out: dict[str, dict] = {}
    scans = {p.name: p for p in realign.SCANS.glob('IMSLP*.pdf')}
    keep = train.reserved()
    orchestral = {}
    if (ORCHESTRA / 'sources.txt').exists():
        for line in (ORCHESTRA / 'sources.txt').read_text().splitlines():
            name, ids = line.split('|')
            for fid in ids.split():
                orchestral.setdefault(fid.lstrip('0'), []).append(pathlib.Path(name).stem)
    songs = yaml.safe_load(LIEDER.read_text()) if LIEDER.exists() else {}
    listed = pathlib.Path(__file__).resolve().parents[2] / 'reserved.yaml'
    pdmx = (yaml.safe_load(listed.read_text()) or {}).get('pdmx') or {} if listed.exists() else {}
    for c in cases.all_cases():
        for t in c.tracks:
            if t.kind != 'scan' or t.pdf not in scans:
                continue
            music = _music(c)
            entry = out.setdefault(t.pdf, {'pdf': scans[t.pdf], 'rows': [], 'music': music,
                                           'staffwise': music == 'orchestra', 'masks': set()})
            entry['masks'] |= c.masks  # families the answer key does not hold as printed
            if not any(r.get('mxl') == str(c.truth_file) for r in entry['rows']):
                entry['rows'].append({'id': c.id, 'path': c.id, 'mxl': str(c.truth_file)})
    for name, pdf in scans.items():
        if not train._is_reserved(name, keep):
            continue
        number = _number(name)
        entry = out.setdefault(name, {'pdf': pdf, 'rows': [], 'music': 'other', 'staffwise': False, 'masks': set()})
        for stem in orchestral.get(number, []):
            mxl = str(ORCHESTRA / 'answers' / f'{stem}.mxl')
            if not any(pathlib.Path(r.get('mxl', '')).stem == stem for r in entry['rows']):
                entry['rows'].append({'id': stem, 'path': stem, 'mxl': mxl})
            entry['music'], entry['staffwise'] = 'orchestra', True
        for sid, song in songs.items():
            if str(song.get('imslp', '')).strip().lstrip('#').lstrip('0') == number:
                if any(pathlib.Path(r.get('mxl', '')).stem == f'lc{sid}' for r in entry['rows']):
                    continue  # a ladder case's own answer key
                entry['rows'].append({'id': str(sid), 'path': song['path'], 'collection': 'lieder'})
                entry['music'] = 'voice'
        made = pdmx.get(int(number)) if number.isdigit() else None
        if made and settings.path('pdmx'):
            for mxl in made.get('mxl', []):  # a PDMX user's transcription of this scan
                entry['rows'].append({'id': pathlib.Path(mxl).stem, 'path': mxl, 'mxl': str(settings.path('pdmx') / mxl)})
            entry['music'] = made.get('music', entry['music'])
    return out


def _music(case) -> str:
    if case.id.startswith(('mozart', 'beethoven-1', 'brahms-3')):
        return 'orchestra'
    if case.id.startswith('arabesque'):
        return 'piano'
    if case.id.startswith('lied'):
        return 'voice'
    return 'quartet' if case.level >= 5 else 'other'


def build(model: pathlib.Path, only: list[str] | None = None) -> dict[str, int]:
    """Pair every reserved scan, reading with [model], and write the test
    sets, or only those of the kinds of music [only] names. Returns how
    many staves each set holds."""
    from . import model as M
    from .progress import Progress
    plans = {k: v for k, v in plan().items() if not only or v['music'] in only}
    net = M.load(model)
    name = f'{model.parent.name}-{model.stem}'
    bar = Progress('testsets', len(plans), 'scans', 'pairing the reserved scans with their transcriptions, unfiltered')
    by_music: dict[str, list[dict]] = {}
    for k, (pdf_name, p) in enumerate(sorted(plans.items()), start=1):
        bar.update(k - 1, pdf_name[:50])
        realign.readings(p['pdf'], net, name)
        out = CACHE / 'data' / f"test-{p['music']}"
        out.mkdir(parents=True, exist_ok=True)
        pair = realign.pair_staves if p['staffwise'] else realign.pair_volume
        samples, report = pair(p['pdf'], p['rows'], net, name, out, _number(pdf_name), keep=0.0, fill=True)
        if p['staffwise']:
            partners(samples)
        rows = _relabel(samples)
        for r in rows:
            r['tokens'] = marks.drop(r['tokens'], p['masks'])
        key_masks(rows, _number(pdf_name))
        rows = as_printed(rows, _number(pdf_name))
        by_music.setdefault(p['music'], []).extend(rows)
        print(json.dumps(report), flush=True)
    if by_music.get('voice'):
        lines = lyric_lines(by_music['voice'], CACHE / 'data' / 'test-lyrics')
        print(json.dumps({'lyric lines': len(lines)}), flush=True)
    counts = {}
    for music, rows in by_music.items():
        with open(CACHE / 'data' / f'test-{music}' / 'index.jsonl', 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        counts[music] = len(rows)
    bar.finish(', '.join(f'test-{m}: {n} staves' for m, n in sorted(counts.items())))
    return counts


def key_masks(rows: list[dict], number: str) -> int:
    """Mark the rows reserved.yaml says the answer does not hold as
    printed: those of a volume listed under unprinted_keys whose instrument
    (or second player's) is one it names leave their key signatures
    unscored, and those of a transcription listed under answer_lacks leave
    the families of marks it names unscored. Returns how many rows were
    marked."""
    from . import instruments
    listed = pathlib.Path(__file__).resolve().parents[2] / 'reserved.yaml'
    found = (yaml.safe_load(listed.read_text()) or {}) if listed.exists() else {}
    named = found.get('unprinted_keys') or {}
    lacks = found.get('answer_lacks') or {}
    kinds = set(named.get(int(number), []) if number.isdigit() else [])
    names, marked = {}, 0
    for r in rows:
        name = re.sub(r'\.(musicxml|mxl|xml)$', '', pathlib.Path(r['mxl'].split(':', 1)[-1]).name)
        masks = set(lacks.get(name, []))  # not Path.stem: 'Brahms_Op.90_4' has no suffix
        if kinds:
            path = _answer(r)
            if path not in names:
                names[path] = [sp.findtext('part-name') or '' for sp in _root(canonical.load(path)).iter('score-part')]
            played = [names[path][k] for k in (r['part'], r.get('with')) if k is not None and k < len(names[path])]
            if any((i := instruments.instrument(n)) is not None and i.name in kinds for n in played):
                masks.add('key')
        if masks:
            r['masks'] = sorted(set(r.get('masks', [])) | masks)
            marked += 1
    return marked


def as_printed(rows: list[dict], number: str) -> list[dict]:
    """The rows written as their edition prints them, by reserved.yaml: a
    part a volume prints on another part's staff throughout (printed_with:
    Mozart's double bass on the cellos' staff) is left out, and a part it
    prints in a plain bass clef sounding an octave lower (plain_bass_clef:
    an old edition's contrabassoon), which the transcription writes in the
    octave bass clef, is relabeled in the plain clef an octave higher. A
    page's own shared parts (a case's) are the ladder's alone: a part that
    rests on one page may have a staff of its own on the next."""
    from . import instruments
    listed = pathlib.Path(__file__).resolve().parents[2] / 'reserved.yaml'
    found = (yaml.safe_load(listed.read_text()) or {}) if listed.exists() else {}
    key = int(number) if number.isdigit() else None
    plain = set((found.get('plain_bass_clef') or {}).get(key, []))
    beside = set((found.get('printed_with') or {}).get(key, []))
    if not plain and not beside:
        return rows
    names, out = {}, []
    for r in rows:
        path = _answer(r)
        if path not in names:
            names[path] = [sp.findtext('part-name') or '' for sp in _root(canonical.load(path)).iter('score-part')]
        name = names[path][r['part']] if r['part'] < len(names[path]) else ''
        if name in beside and r.get('with') is None:
            continue
        inst = instruments.instrument(name)
        if inst is not None and inst.name in plain and 'clef:F4-8' in r['tokens']:
            r = {**r, 'tokens': [_octave_up(t) for t in r['tokens']]}
        out.append(r)
    return out


def _octave_up(t: str) -> str:
    """A token with its pitch an octave higher, and the octave bass clef plain."""
    if t == 'clef:F4-8':
        return 'clef:F4'
    m = re.match(r'(N:[A-G])(-?\d+)$', t)
    return f'{m.group(1)}{int(m.group(2)) + 1}' if m else t


def partners(rows: list[dict]) -> int:
    """A numbered player whose partner (1 and 2, 3 and 4) has no staff of
    its own in the system is printed on this staff with it, or rests and
    is not printed: the row is labeled with both players, as pairing labels
    the two-player staves it finds (tokens.merged_part keeps a resting
    partner silent). Returns how many rows were given a partner."""
    names: dict = {}
    systems: dict = collections.defaultdict(set)

    def system(r):
        return r['pdf'], re.sub(r'-n\d+\.png$', '', r['image'])
    for r in rows:
        systems[system(r)] |= {r['part']} | ({r['with']} if r.get('with') is not None else set())
    given = 0
    for r in rows:
        if r.get('with') is not None:
            continue
        path = _answer(r)
        if path not in names:
            names[path] = [sp.findtext('part-name') or '' for sp in _root(canonical.load(path)).iter('score-part')]
        ns = names[path]
        m = re.match(r'(.*?)\s*(\d)$', ns[r['part']] if r['part'] < len(ns) else '')
        if not m:
            continue
        n = int(m.group(2))
        other = next((j for j, x in enumerate(ns) if x == f'{m.group(1)} {n + 1 if n % 2 else n - 1}'), None)
        if other is None or other in systems[system(r)]:
            continue
        r['part'], r['with'] = min(r['part'], other), max(r['part'], other)
        given += 1
    return given


def _answer(r: dict) -> pathlib.Path:
    stem = r['mxl'].split(':', 1)[1] if r['mxl'].startswith('openscore:') else None
    return (ANSWERS / f'{stem}.mxl' if stem and (ANSWERS / f'{stem}.mxl').exists()
            else ORCHESTRA / 'answers' / f'{stem}.mxl' if stem else pathlib.Path(r['mxl']))


def lyric_lines(samples: list[dict], out: pathlib.Path, kind: str = 'lyrics') -> list[dict]:
    """Each verse sung under a paired staff, cut from its page as the
    reader cuts it (lyrics.rows) and labeled with its syllables as the
    rendered lines are (data.verse_text). A staff is used only where the
    rows found under it are as many as its verses, so a row is never
    labeled with another verse's words."""
    from . import layout, lyrics, real
    from .data import _verses, verse_text
    lines, parts, level, at = [], {}, None, None
    (out / 'text').mkdir(parents=True, exist_ok=True)
    for r in samples:
        if 'at' not in r:
            continue
        path = _answer(r)
        if path not in parts:
            parts[path] = [p for p, _ in _parts(_root(canonical.load(path)))]
        part = parts[path][r['part']]
        verses = _verses(part, r['staff'], r['first'], r['last'])
        if not verses:
            continue
        png = real.pages(real.SCANS / r['pdf'])[r['page'] - 1]
        found = real._found(png)
        if at != png:
            level, at = layout.level(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE), found['angle'], found.get('bend')), png
        st = found['staves'][r['at']]
        staff = layout.Staff(st['lines'], st['x0'], st['x1'])
        below = found['staves'][r['at'] + 1]['lines'][0] if r['at'] + 1 < len(found['staves']) else level.shape[0]
        rows = lyrics.rows(level, staff, below)
        if len(rows) != len(verses):
            continue
        pad = int(0.4 * staff.space)
        for verse, (a, b) in zip(verses, rows):
            text = ' '.join(verse_text(part, r['staff'], r['first'], r['last'], verse))
            file = out / 'text' / f"{pathlib.Path(r['pdf']).stem.split('-')[0]}-{pathlib.Path(r['image']).stem}-v{verse}.png"
            cv2.imwrite(str(file), level[max(0, a - pad):b + pad, int(staff.x0):int(staff.x1) + 1])
            lines.append({'image': str(file.relative_to(out)), 'text': text, 'kind': kind, 'font': 'scan'})
    with open(out / 'text' / 'lines.jsonl', 'w') as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')
    return lines


def _relabel(samples: list[dict]) -> list[dict]:
    """Format-2 tokens for each paired staff, a system's first staff as
    its top."""
    out, cache = [], {}
    for r in samples:
        place = re.search(r'-n(\d+)\.png$', r['image'])
        top = int(place.group(1)) == 0 if place else (r['part'], r['staff']) == (0, 1)
        key = (r['mxl'], r['part'], r['staff'], top, r.get('with'))
        if key not in cache:
            parts = [p for p, _ in _parts(_root(canonical.load(_answer(r))))]
            if r.get('with') is not None:  # two players printed on one staff
                cache[key] = tokens.from_part(tokens.merged_part(parts[r['part']], parts[r['with']]), 1, top=top)
            else:
                cache[key] = tokens.from_part(parts[r['part']], r['staff'], top=top)
        bars, states = cache[key]
        if r['last'] < len(bars) and all(s['readable'] for s in states[r['first']:r['last'] + 1]):
            out.append({**r, 'top': top, 'tokens': tokens.stretch(bars, states, r['first'], r['last']).tokens})
    return out
