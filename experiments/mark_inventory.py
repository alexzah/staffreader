"""Every kind of mark MusicXML carries, counted in the music we read:
the corpus's score bundles, a sample of the PDMX scores we render, and
OpenScore's quartet, Lieder and orchestral transcriptions. For each mark,
how many scores of each source use it and how often it occurs. Writes
inventory/marks.json."""
import collections, json, pathlib, random, sys
from concurrent.futures import ProcessPoolExecutor
from staffreader import canonical, settings
from staffreader.data import PDMX, _root
from staffreader.render import CACHE

SCORES = settings.path('scores')


def marks(path) -> collections.Counter:
    c = collections.Counter()
    try:
        root = _root(canonical.load(path))
    except Exception:
        return c
    for note in root.iter('note'):
        for tag in ('grace', 'cue'):
            if note.find(tag) is not None:
                c[f'note: {tag}'] += 1
        for ly in note.findall('lyric'):
            c['lyrics: syllable'] += 1
            if ly.get('number', '1') not in ('1', None):
                c['lyrics: verse 2+'] += 1
            if ly.findtext('syllabic') in ('begin', 'middle'):
                c['lyrics: hyphen'] += 1
            if ly.find('extend') is not None:
                c['lyrics: extender'] += 1
        nh = note.findtext('notehead')
        if nh and nh != 'normal':
            c[f'notehead: {nh}'] += 1
        acc = note.find('accidental')
        if acc is not None and (acc.get('parentheses') == 'yes' or acc.get('editorial') == 'yes' or acc.get('bracket') == 'yes'):
            c['accidental: editorial or cautionary'] += 1
        for n in note.findall('notations'):
            for child in n:
                if child.tag in ('articulations', 'ornaments', 'technical'):
                    for g in child:
                        if g.tag != 'accidental-mark':
                            c[f'{child.tag}: {g.tag}'] += 1
                        else:
                            c['ornaments: accidental-mark'] += 1
                elif child.tag == 'dynamics':
                    c['dynamics (on a note)'] += 1
                else:
                    c[f'notations: {child.tag}'] += 1
    for d in root.iter('direction'):
        for dt in d.findall('direction-type'):
            for child in dt:
                if child.tag == 'dynamics':
                    for g in child:
                        c[f'dynamics: {g.tag if g.tag != "other-dynamics" else (g.text or "other")}'] += 1
                elif child.tag == 'octave-shift' and child.get('type') != 'stop':
                    c[f'octave-shift: {child.get("size", "8")}'] += 1
                elif child.tag in ('wedge', 'dashes', 'bracket', 'pedal') and child.get('type') in ('stop', 'continue'):
                    continue
                else:
                    c[f'direction: {child.tag}'] += 1
    for s in root.iter('sound'):
        for a in ('dacapo', 'dalsegno', 'fine', 'tocoda', 'segno', 'coda'):
            if s.get(a):
                c[f'jump: {a}'] += 1
    for b in root.iter('barline'):
        r = b.find('repeat')
        if r is not None:
            c[f'barline: repeat {r.get("direction")}'] += 1
        e = b.find('ending')
        if e is not None and e.get('type') == 'start':
            c['barline: ending (volta)'] += 1
        st = b.findtext('bar-style')
        if st and st not in ('regular', 'light-heavy', 'light-light', 'heavy-light'):
            c[f'barline: {st}'] += 1
        for t in ('segno', 'coda', 'fermata'):
            if b.find(t) is not None:
                c[f'barline: {t}'] += 1
    for a in root.iter('attributes'):
        for ms in a.findall('measure-style'):
            for g in ms:
                c[f'measure-style: {g.tag}'] += 1
        if a.find('transpose') is not None:
            c['attributes: transposing part'] += 1
        for cl in a.findall('clef'):
            sign, line, octv = cl.findtext('sign'), cl.findtext('line'), cl.findtext('clef-octave-change')
            c[f'clef: {sign}{line or ""}{("+" if octv and int(octv) > 0 else "") + octv if octv else ""}'] += 1
        for t in a.findall('time'):
            if t.find('senza-misura') is not None:
                c['time: senza misura'] += 1
            elif '+' in (t.findtext('beats') or ''):
                c['time: compound (2+3)'] += 1
    for tag in ('harmony', 'figured-bass'):
        n = sum(1 for _ in root.iter(tag))
        if n:
            c[tag] += n
    return c


def sources() -> dict[str, list]:
    random.seed(0)
    out = {'corpus': sorted(SCORES.glob('bundles/*/score.musicxml')),
           'quartets': sorted((CACHE / 'openscore/answers').glob('sq*.mxl')),
           'lieder': sorted((CACHE / 'openscore/answers').glob('lc*.mxl')),
           'orchestra': sorted((CACHE / 'openscore/orchestra/answers').glob('*.mxl'))}
    rows = [json.loads(l) for l in (CACHE / 'data/scores.jsonl').read_text().splitlines()]
    out['pdmx'] = [PDMX / r['mxl'] for r in random.sample(rows, 2000)]
    return out


if __name__ == '__main__':
    result = {}
    with ProcessPoolExecutor(12) as pool:
        for name, paths in sources().items():
            counts = list(pool.map(marks, paths, chunksize=8))
            used = collections.Counter(k for c in counts for k in c)
            total = sum(counts, collections.Counter())
            result[name] = {'scores': len(paths), 'used': dict(used), 'total': dict(total)}
            print(name, len(paths), 'scores', len(used), 'kinds of mark', flush=True)
    (CACHE / 'inventory').mkdir(exist_ok=True)
    (CACHE / 'inventory/marks.json').write_text(json.dumps(result, indent=1))
