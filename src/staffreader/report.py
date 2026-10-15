"""The scoreboard: every case and track against every reader, with the
headline numbers in a table and, for each run, which bars were read right."""

from __future__ import annotations

import html
import json
import os
import pathlib
import shutil

from . import cases as cases_mod

COLUMNS = [
    ('note_f1_core', 'Notes (pitch + onset) F1'),
    ('note_f1_rhythm', '+ length'),
    ('bar_exact_core', 'Bars right (notes)'),
    ('bar_exact_full', 'Bars right (all)'),
    ('rest_f1', 'Rests F1'),
    ('attributes', 'Clef/key/time'),
    ('structure', 'Repeats/endings'),
    ('bar_valid', 'Read bars that add up'),
]

STYLE = """
:root { color-scheme: light dark; --bg: #fbfaf7; --fg: #1d1d1b; --muted: #6b6a66; --line: #dddad3;
        --good: #1f7a4d; --mid: #9a6a00; --bad: #b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ecebe6; --muted: #a3a29c; --line: #34332f;
        --good: #5cc48f; --mid: #e0b44a; --bad: #f2867d; } }
body { background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, sans-serif; margin: 0 auto;
       max-width: 1200px; padding: 24px 16px; }
h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 28px; } .muted { color: var(--muted); }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 5px 8px; text-align: right; vertical-align: top; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) { text-align: left; }
.good { color: var(--good); } .mid { color: var(--mid); } .bad { color: var(--bad); }
.bars span { display: inline-block; width: 14px; height: 14px; margin: 1px; border-radius: 2px; }
.bars .ok { background: var(--good); } .bars .notes { background: var(--mid); } .bars .no { background: var(--bad); }
.bars .gap { background: transparent; outline: 1px dashed var(--bad); }
img.page { max-width: 100%; border: 1px solid var(--line); }
"""


def _cls(v) -> str:
    if not isinstance(v, (int, float)):
        return ''
    return 'good' if v >= 0.97 else ('mid' if v >= 0.85 else 'bad')


def _cell(v) -> str:
    if v is None:
        return '<td class="muted">—</td>'
    if isinstance(v, float):
        return f'<td class="{_cls(v)}">{v * 100:.1f}%</td>'
    return f'<td>{html.escape(str(v))}</td>'


def write(results_json: pathlib.Path, out: pathlib.Path):
    results = json.loads(results_json.read_text())
    by_case = {}
    for r in results:
        by_case.setdefault((r['level'], r['case']), []).append(r)
    rows = []
    details = []
    for (level, case_id), rs in sorted(by_case.items()):
        case = cases_mod.load(cases_mod.CASES / f'{case_id}.yaml')
        for r in rs:
            s = r['summary']
            rows.append('<tr>' + f'<td>L{level}</td><td><a href="#{case_id}">{html.escape(case_id)}</a></td>'
                        f'<td>{html.escape(r["track"])} · {html.escape(r["reader"])}</td>'
                        + ''.join(_cell(s.get(k)) for k, _ in COLUMNS)
                        + f'<td>{s["measures_missed"]}/{s["measures_extra"]}</td><td>{r["seconds"]}s</td></tr>')
        details.append(_detail(case, rs, out))
    head = ''.join(f'<th>{html.escape(t)}</th>' for _, t in COLUMNS)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>OMR scoreboard</title>
<style>{STYLE}</style></head><body>
<h1>OMR scoreboard</h1>
<p class="muted">Each reading compared with the score we already hold. Bars are lined up first, so a missed
barline costs one bar. "Bars right (all)" is the share of bars needing no correction.</p>
<table><tr><th>Level</th><th>Case</th><th>Track · reader</th>{head}<th>Bars missed/extra</th><th>Time</th></tr>
{''.join(rows)}</table>
{''.join(details)}
</body></html>"""
    (out / 'index.html').write_text(page)
    return out / 'index.html'


def _detail(case, rs, out: pathlib.Path) -> str:
    parts = []
    for r in rs:
        legend = []
        by_part = {}
        for b in r['bars']:
            by_part.setdefault(b['part'], []).append(b)
        lines = []
        for p, bars in sorted(by_part.items()):
            cells = []
            for b in bars:
                if b['gt'] is None:
                    cells.append('<span class="gap" title="extra bar read"></span>')
                    continue
                cls = 'ok' if b['exact_full'] else ('notes' if b['exact_core'] else 'no')
                if b['pred'] is None:
                    cls = 'gap'
                tip = f"bar {b['number']}: {b['notes']} notes, {b['missed']} missed, {b['extra']} extra"
                cells.append(f'<span class="{cls}" title="{html.escape(tip)}"></span>')
            lines.append(f'<div class="bars">part {p + 1} {"".join(cells)}</div>')
        err = f'<p class="bad">{html.escape(r["error"])}</p>' if r.get('error') else ''
        kinds = r['summary'].get('errors') or {}
        if kinds:
            err += '<p class="muted">Mistakes: ' + ', '.join(f'{html.escape(k)} {v}' for k, v in kinds.items()) + '</p>'
        links = ' '.join(f'<a href="{html.escape(_copy(x, out, case.id, r))}">reading</a>' for x in r['readings'])
        parts.append(f'<h3>{html.escape(r["track"])} · {html.escape(r["reader"])}</h3>{err}{"".join(lines)}<p>{links}</p>')
    pages = ''
    for t in case.tracks:
        for image, page in case.pages(t):
            src = _preview(image, page, out, case.id, t.name)
            pages += f'<p class="muted">{html.escape(t.name)}</p><img class="page" loading="lazy" src="{src}" alt="">'
    return (f'<h2 id="{case.id}">L{case.level} · {html.escape(case.title)}</h2>'
            f'<p class="muted">{html.escape(case.proves)}. Green: bar right in full; amber: notes right, '
            f'something else wrong; red: notes wrong; dashed: bar missing or extra.</p>'
            + ''.join(parts) + f'<details><summary>Pages</summary>{pages}</details>')


def _copy(xml: str, out: pathlib.Path, case: str, r) -> str:
    target = out / 'readings' / case / f'{r["track"]}-{r["reader"]}-{pathlib.Path(xml).name}'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(xml, target)
    return os.path.relpath(target, out)


def _preview(image: pathlib.Path, page, out: pathlib.Path, case: str, track: str) -> str:
    import subprocess
    target = out / 'pages' / case / f'{track}-{image.stem}.png'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if page is None:
            subprocess.run(['magick', str(image), '-resize', '1000x', str(target)], check=True)
        else:
            subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-r', '110', '-gray', '-singlefile',
                            '-png', str(image), str(target.with_suffix(''))], check=True)
    return os.path.relpath(target, out)
