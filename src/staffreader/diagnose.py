"""Where a reader is weak and where it is strong (ITERATING.md, steps 2
and 3).

`measure` writes a record for each staff it reads: the staff's answer,
its reading and their scores. This module slices those records by what
each staff holds and where it came from, scores every slice with a 90%
bootstrap interval, ranks the slices by their cost to the reader, and
draws the worst staves of the weakest slices with their answer and
reading side by side. Given a second model's records of the same staves,
it compares the two slice by slice and marks what got worse beyond the
noise."""

from __future__ import annotations

import base64
import collections
import difflib
import html
import json
import pathlib
import random

import cv2

FAMILY_OF_SET = [('real', 'quartet'), ('lieder', 'voice and piano'), ('orch', 'orchestra'), ('r2-grand', 'piano'),
                 ('m4-a', 'piano'), ('r2-tenor', 'tenor clef'), ('m5-tenor', 'tenor clef'), ('r2-single', 'one staff'),
                 ('m3-a', 'one staff'), ('test-', 'reserved')]
SLICES = ['music', 'source', 'set', 'clef', 'key', 'time', 'density', 'voices', 'chords', 'tuplets', 'grace',
          'marks']


def music_of(set_name: str) -> str:
    for prefix, music in FAMILY_OF_SET:
        if set_name.startswith(prefix):
            return music if prefix != 'test-' else set_name.removeprefix('test-')
    return 'other'


def features(record: dict) -> dict[str, str]:
    """What a staff holds and where it came from, as the slices name it."""
    marks = sorted({f for f, (tp, fp, fn) in record.get('marks', {}).items() if tp + fn})
    return {'music': music_of(record['set']), 'source': record['kind'], 'set': record['set'],
            **token_features(record['truth']), 'marks': ', '.join(marks[:3]) if marks else 'no marks'}


def token_features(toks: list[str]) -> dict[str, str]:
    """What a staff's tokens show, as the slices name it: the features a
    training run can be told to show more of (train.samples, boost)."""
    notes = sum(t.startswith('N:') for t in toks)
    bars = max(1, sum(t in ('|', '||', '|]', ':|', '|:', ':|:') for t in toks))
    density = notes / bars
    key = next((int(t[4:]) for t in toks if t.startswith('key:')), 0)
    return {
        'clef': next((t[5:] for t in toks if t.startswith('clef:')), '?'),
        'key': '0' if key == 0 else ('1-2' if abs(key) <= 2 else ('3-4' if abs(key) <= 4 else '5-7')) + (' flats' if key < 0 else ' sharps'),
        'time': next((t[5:] for t in toks if t.startswith('time:')), 'not printed'),
        'density': '<4 a bar' if density < 4 else ('4-8 a bar' if density < 8 else ('8-16 a bar' if density < 16 else '16+ a bar')),
        'voices': 'two or more' if '&' in toks else 'one',
        'chords': 'chords' if '+' in toks else 'no chords',
        'tuplets': 'tuplets' if any(t.startswith('tup:') for t in toks) else 'no tuplets',
        'grace': 'grace notes' if 'gr' in toks else 'no grace notes',
    }


def edits(record: dict) -> int:
    """The corrections a person would make to the staff's reading, roughly:
    one for each wrong, missed or extra note or rest, each mark wrong or
    missing, and one for a wrong clef or key, whose consequences (the
    pitches it moves) are not counted again."""
    e = dict(record.get('errors', {}))
    count = sum(e.values())
    if e.get('clef printed wrong or missed') or e.get('key printed wrong or missed'):
        count -= e.get('wrong pitch', 0) + e.get('wrong accidental', 0)
    count += sum(fp + fn for tp, fp, fn in record.get('marks', {}).values())
    return max(0, count)


def _f1(tp: int, fp: int, fn: int) -> float:
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return 2 * p * r / max(1e-9, p + r)


def score(records: list[dict]) -> dict:
    """The scorecard of a group of staves."""
    ok = [r for r in records if not r.get('unreadable')]
    tp, fp, fn = (sum(r.get(k, 0) for r in ok) for k in ('tp', 'fp', 'fn'))
    bars, exact = sum(r['bars'] for r in ok), sum(r['bars_exact'] for r in ok)
    fam: dict[str, list[int]] = {}
    for r in ok:
        for f, t in r.get('marks', {}).items():
            acc = fam.setdefault(f, [0, 0, 0])
            for k in range(3):
                acc[k] += t[k]
    return {'staves': len(records), 'unreadable': len(records) - len(ok), 'note_f1': _f1(tp, fp, fn),
            'bars_exact': exact / max(1, bars), 'edits_per_staff': sum(edits(r) for r in ok) / max(1, len(ok)),
            'marks': {f: _f1(*t) for f, t in fam.items() if t[0] + t[2]}}


def interval(records: list[dict], measure: str, rounds: int = 200, seed: int = 0) -> tuple[float, float]:
    """A 90% bootstrap interval of a measure over a group of staves."""
    if len(records) < 2:
        v = score(records)[measure]
        return v, v
    rng = random.Random(seed)
    values = sorted(score([rng.choice(records) for _ in records])[measure] for _ in range(rounds))
    return values[int(0.05 * rounds)], values[int(0.95 * rounds) - 1]


def slices(records: list[dict]) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {s: collections.defaultdict(list) for s in SLICES}
    for r in records:
        for name, value in features(r).items():
            out[name][value].append(r)
    return out


def impact(records: list[dict], minimum: int = 20) -> list[dict]:
    """Each slice's cost to the reader: how far its bars read exactly fall
    below all staves', times its share of the staves. Largest first."""
    whole = score(records)['bars_exact']
    out = []
    for name, groups in slices(records).items():
        for value, group in groups.items():
            if len(group) < minimum or name == 'set':
                continue
            s = score(group)
            out.append({'slice': name, 'value': value, 'staves': len(group), 'bars_exact': s['bars_exact'],
                        'note_f1': s['note_f1'], 'edits_per_staff': s['edits_per_staff'],
                        'interval': interval(group, 'bars_exact'),
                        'impact': (whole - s['bars_exact']) * len(group) / len(records)})
    return sorted(out, key=lambda x: -x['impact'])


def compare(before: list[dict], after: list[dict], minimum: int = 20) -> list[dict]:
    """Each slice's change from [before] to [after] on the staves both
    read, with a paired 90% bootstrap interval; a change whose interval
    lies wholly below zero is a regression, wholly above an improvement."""
    by_path = {r['path']: r for r in before}
    pairs = [(by_path[r['path']], r) for r in after if r['path'] in by_path]
    out = []
    groups: dict[tuple[str, str], list[tuple[dict, dict]]] = collections.defaultdict(list)
    for b, a in pairs:
        groups[('all', 'all')].append((b, a))
        for name, value in features(a).items():
            if name != 'set':
                groups[(name, value)].append((b, a))
    rng = random.Random(0)
    for (name, value), group in groups.items():
        if len(group) < minimum:
            continue
        delta = score([a for _, a in group])['bars_exact'] - score([b for b, _ in group])['bars_exact']
        draws = []
        for _ in range(200):
            pick = [rng.choice(group) for _ in group]
            draws.append(score([a for _, a in pick])['bars_exact'] - score([b for b, _ in pick])['bars_exact'])
        draws.sort()
        low, high = draws[10], draws[189]
        out.append({'slice': name, 'value': value, 'staves': len(group), 'change': delta, 'interval': (low, high),
                    'verdict': 'worse' if high < 0 else ('better' if low > 0 else 'no clear change')})
    return sorted(out, key=lambda x: x['change'])


def _picture(path: str, width: int = 900) -> str:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ''
    scale = min(1.0, width / img.shape[1])
    img = cv2.resize(img, None, fx=scale, fy=scale)
    ok, png = cv2.imencode('.png', img)
    return base64.b64encode(png.tobytes()).decode() if ok else ''


def _diff(truth: list[str], read: list[str]) -> str:
    out = []
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, truth, read, autojunk=False).get_opcodes():
        if op == 'equal':
            out.append(html.escape(' '.join(truth[a0:a1])))
        else:
            if a1 > a0:
                out.append(f"<del>{html.escape(' '.join(truth[a0:a1]))}</del>")
            if b1 > b0:
                out.append(f"<ins>{html.escape(' '.join(read[b0:b1]))}</ins>")
    return ' '.join(out)


def gallery(group: list[dict], count: int = 6) -> list[dict]:
    """The worst staves of a group: fewest bars read exactly, then most
    edits."""
    ok = [r for r in group if not r.get('unreadable')]
    worst = sorted(ok, key=lambda r: (r['bars_exact'] / max(1, r['bars']), -edits(r)))[:count]
    return [{'path': r['path'], 'picture': _picture(r['path']), 'diff': _diff(r['truth'], r['read']),
             'edits': edits(r), 'bars': f"{r['bars_exact']}/{r['bars']}"} for r in worst]


def ladder(results: pathlib.Path) -> dict[str, dict]:
    """A ladder run's real-scan pages by kind of music, from its results."""
    kinds = [('arabesque', 'solo piano'), ('lied', 'voice and piano'), ('mozart', 'orchestra'),
             ('beethoven-1', 'orchestra'), ('brahms-3', 'orchestra'), ('paganini', 'one staff'), ('petzold', 'piano')]
    out: dict[str, list] = collections.defaultdict(list)
    for r in json.loads(results.read_text()):
        if not r.get('summary') or not r['track'].startswith('scan'):
            continue
        kind = next((k for p, k in kinds if r['case'].startswith(p)), 'quartet')
        out[kind].append(r)
    return {k: {'pages': len(v), 'note_f1': sum(r['summary']['note_f1_core'] for r in v) / len(v),
                'bars_exact': sum(r['summary']['bar_exact_full'] for r in v) / len(v)} for k, v in out.items()}


def read_records(paths: list[pathlib.Path]) -> list[dict]:
    out = []
    for p in paths:
        for line in open(p):
            out.append(json.loads(line))
    return out


def report(records: list[dict], out: pathlib.Path, title: str, against: list[dict] | None = None,
           ladder_runs: dict[str, pathlib.Path] | None = None, weak: int = 6, text: dict | None = None) -> pathlib.Path:
    """The diagnosis as a page, out/index.html, and its numbers,
    out/diagnosis.json."""
    out.mkdir(parents=True, exist_ok=True)
    whole = score(records)
    ranked = impact(records)
    groups = slices(records)
    by_set = {k: score(v) for k, v in groups['set'].items()}
    changes = compare(against, records) if against else []
    ladders = {name: ladder(p) for name, p in (ladder_runs or {}).items() if p.exists()}
    (out / 'diagnosis.json').write_text(json.dumps({
        'title': title, 'overall': whole, 'sets': by_set, 'impact': ranked, 'changes': changes, 'ladder': ladders,
        'text': text},
        indent=1, default=list))
    pct = lambda v: f'{100 * v:.1f}%'
    rows = ''.join(f"<tr><td>{html.escape(k)}</td><td>{v['staves']}</td><td>{v['note_f1']:.3f}</td>"
                   f"<td>{pct(v['bars_exact'])}</td><td>{v['edits_per_staff']:.2f}</td></tr>" for k, v in sorted(by_set.items()))
    weak_rows = ''.join(
        f"<tr><td>{x['slice']}</td><td>{html.escape(x['value'])}</td><td>{x['staves']}</td><td>{pct(x['bars_exact'])} "
        f"<small>({pct(x['interval'][0])}–{pct(x['interval'][1])})</small></td><td>{x['note_f1']:.3f}</td>"
        f"<td>{x['edits_per_staff']:.2f}</td><td>{100 * x['impact']:.2f}</td></tr>" for x in ranked[:20])
    strong_rows = ''.join(
        f"<tr><td>{x['slice']}</td><td>{html.escape(x['value'])}</td><td>{x['staves']}</td><td>{pct(x['bars_exact'])}</td></tr>"
        for x in sorted(ranked, key=lambda x: -x['bars_exact'])[:10])
    change_rows = ''.join(
        f"<tr class='{x['verdict'].split()[0]}'><td>{x['slice']}</td><td>{html.escape(x['value'])}</td><td>{x['staves']}</td>"
        f"<td>{100 * x['change']:+.1f}</td><td>{100 * x['interval'][0]:+.1f} to {100 * x['interval'][1]:+.1f}</td>"
        f"<td>{x['verdict']}</td></tr>" for x in changes)
    text_rows = ''.join(f"<tr><td>{html.escape(k)}</td><td>{v['lines']}</td><td>{pct(v['exact'])}</td>"
                        f"<td>{pct(v['cer'])}</td><td>{', '.join(f'{a}: {pct(b)}' for a, b in v.get('by_kind', {}).items())}</td></tr>"
                        for k, v in (text or {}).items())
    marks_rows = ''.join(f"<tr><td>{html.escape(f)}</td><td>{v:.3f}</td></tr>" for f, v in sorted(whole['marks'].items()))
    ladder_rows = ''.join(f"<tr><td>{html.escape(name)}</td><td>{html.escape(k)}</td><td>{v['pages']}</td>"
                          f"<td>{v['note_f1']:.3f}</td><td>{pct(v['bars_exact'])}</td></tr>"
                          for name, kinds in ladders.items() for k, v in sorted(kinds.items()))
    shows = []
    for x in ranked[:weak]:
        for g in gallery(groups[x['slice']][x['value']], 3):
            shows.append(f"<figure><figcaption>{x['slice']}: {html.escape(x['value'])} · bars {g['bars']} exact · "
                         f"{g['edits']} edits · {html.escape(pathlib.Path(g['path']).name)}</figcaption>"
                         f"<img src='data:image/png;base64,{g['picture']}'><p class='diff'>{g['diff']}</p></figure>")
    page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse;margin:.5em 0 1.5em}}td,th{{padding:3px 10px;border-bottom:1px solid #ddd;text-align:left}}
del{{background:#fdd;text-decoration:none}}ins{{background:#dfd;text-decoration:none}}
tr.worse td{{color:#a00}}tr.better td{{color:#070}}figure{{margin:1.5em 0}}img{{max-width:100%;border:1px solid #ccc}}
.diff{{font:12px/1.5 ui-monospace,monospace;word-break:break-word}}small{{color:#666}}</style>
<h1>{html.escape(title)}</h1>
<p>{whole['staves']} staves: note F1 {whole['note_f1']:.3f}, bars exact {pct(whole['bars_exact'])}, {whole['edits_per_staff']:.2f} edits a staff.</p>
<h2>By test set</h2><table><tr><th>Set</th><th>Staves</th><th>Note F1</th><th>Bars exact</th><th>Edits a staff</th></tr>{rows}</table>
{'<h2>The ladder, real scans</h2><table><tr><th>Run</th><th>Music</th><th>Pages</th><th>Note F1</th><th>Bars exact</th></tr>' + ladder_rows + '</table>' if ladder_rows else ''}
<h2>Weakest slices, by cost</h2><p>Cost: how far a slice's bars read exactly fall below all staves', times its share of them, in points.</p>
<table><tr><th>Slice</th><th>Value</th><th>Staves</th><th>Bars exact (90%)</th><th>Note F1</th><th>Edits a staff</th><th>Cost</th></tr>{weak_rows}</table>
<h2>Strongest slices</h2><table><tr><th>Slice</th><th>Value</th><th>Staves</th><th>Bars exact</th></tr>{strong_rows}</table>
{'<h2>Against the model before</h2><table><tr><th>Slice</th><th>Value</th><th>Staves</th><th>Change, points</th><th>90% interval</th><th>Verdict</th></tr>' + change_rows + '</table>' if change_rows else ''}
{'<h2>Marks</h2><table><tr><th>Family</th><th>F1</th></tr>' + marks_rows + '</table>' if marks_rows else ''}
{'<h2>Text</h2><table><tr><th>Lines</th><th>Read</th><th>Exact</th><th>Characters wrong</th><th>Exact by kind</th></tr>' + text_rows + '</table>' if text_rows else ''}
<h2>The worst staves of the weakest slices</h2><p>The answer's tokens, with what the reading <del>left out or got wrong</del> and <ins>wrote instead</ins>.</p>
{''.join(shows)}"""
    (out / 'index.html').write_text(page)
    return out / 'index.html'
