"""The layout stage on the ladder: every page of every case, its staves,
systems and barlines found and compared with the right layout, exact for
a synthetic page and as counted for a scan, with each page drawn over so
a person can see what was found."""

from __future__ import annotations

import html
import json
import os
import pathlib
import subprocess
import tempfile
import time

import cv2

from . import layout, layout_truth
from .cases import Case
from .report import STYLE, _cell

COLUMNS = [('staves_f1', 'Staves'), ('systems_exact', 'Systems right'), ('barlines_f1', 'Barlines'),
           ('bars_per_system_exact', 'Systems with the right bars')]


def gray_page(image: pathlib.Path, page: int | None):
    if page is None:
        return cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    with tempfile.TemporaryDirectory(dir=layout._scratch()) as d:
        out = pathlib.Path(d) / 'page'
        subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-r', '300', '-gray', '-singlefile', '-png',
                        str(image), str(out)], check=True)
        return cv2.imread(str(out) + '.png', cv2.IMREAD_GRAYSCALE)


def run(cases: list[Case], out: pathlib.Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        for track in case.tracks:
            for k, (image, page) in enumerate(case.pages(track)):
                truth = case.layout_truth(track, k)
                if truth is None:
                    continue
                started = time.time()
                gray = gray_page(image, page)
                found = layout.read(gray)
                seconds = round(time.time() - started, 1)
                score = (layout_truth.compare(truth, found) if isinstance(truth, layout.Layout)
                         else layout_truth.compare_counted(truth, found))
                picture = out / 'pages' / f'{case.id}-{track.name}-{k + 1}.png'
                picture.parent.mkdir(parents=True, exist_ok=True)
                layout.draw(gray, found, picture)
                subprocess.run(['magick', str(picture), '-resize', '1000x', str(picture)], check=True)
                r = {'case': case.id, 'level': case.level, 'track': track.name, 'page': k + 1,
                     'summary': score.summary(), 'seconds': seconds, 'picture': os.path.relpath(picture, out),
                     'found': [[len(s.staves), s.measures] for s in found.systems],
                     'truth': ([[len(s.staves), s.measures] for s in truth.systems]
                               if isinstance(truth, layout.Layout) else [list(s) for s in truth.systems])}
                s = r['summary']
                print(f"{case.id:22} {track.name:16} p{k + 1}  staves {s['staves_f1']:.3f}  systems {s['systems_exact']:.3f}  "
                      f"barlines {s['barlines_f1']:.3f}  bars/system {s['bars_per_system_exact']:.3f}  {seconds}s", flush=True)
                results.append(r)
    (out / 'layout.json').write_text(json.dumps(results, indent=1))
    write(results, out)
    return results


def write(results: list[dict], out: pathlib.Path) -> pathlib.Path:
    rows, pics = [], []
    for r in results:
        s = r['summary']
        notes = '; '.join(s.get('notes', []))
        rows.append(f'<tr><td>L{r["level"]}</td><td><a href="#{r["case"]}-{r["track"]}-{r["page"]}">'
                    f'{html.escape(r["case"])}</a></td><td>{html.escape(r["track"])} p{r["page"]}</td>'
                    + ''.join(_cell(s[k]) for k, _ in COLUMNS)
                    + f'<td>{html.escape(str(r["found"]))}</td><td>{html.escape(str(r["truth"]))}</td>'
                    f'<td class="muted">{html.escape(notes)}</td><td>{r["seconds"]}s</td></tr>')
        pics.append(f'<h3 id="{r["case"]}-{r["track"]}-{r["page"]}">{html.escape(r["case"])} · '
                    f'{html.escape(r["track"])} p{r["page"]}</h3><img class="page" loading="lazy" '
                    f'src="{html.escape(r["picture"])}" alt="">')
    head = ''.join(f'<th>{html.escape(t)}</th>' for _, t in COLUMNS)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>OMR layout</title>
<style>{STYLE}</style></head><body>
<h1>OMR layout</h1>
<p class="muted">Staves, systems and barlines found on each page. A synthetic page's answer is exact, from the
engraving; a scan's is counted by eye: each system's staves and bars, as [staves, bars]. On the pages below, staves
are blue, systems are bracketed in green with their number of bars, and barlines are red.</p>
<table><tr><th>Level</th><th>Case</th><th>Page</th>{head}<th>Found</th><th>Right</th><th>Notes</th><th>Time</th></tr>
{''.join(rows)}</table>
{''.join(pics)}
</body></html>"""
    (out / 'index.html').write_text(page)
    return out / 'index.html'
