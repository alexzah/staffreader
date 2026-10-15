"""Real lyric lines for training the text reader: every verse under the
voice staves the Lieder pairing kept (data/lieder-d-v2), cut as the reader
cuts them (testsets.lyric_lines), from every volume but the reserved ones.
Written to data/lieder-lyrics as kind 'lyrics-real', so the text reader
draws them as often as rendered lyrics and names.

The pairing did not record which staff of the page a row came from; in a
song the voice is its system's first staff, and lyric_lines keeps a staff
only where the rows under it are as many as its verses."""

import json
import re
import sys

from staffreader import testsets, train
from staffreader.progress import Progress
from staffreader.render import CACHE
from staffreader import real

rows = [json.loads(l) for l in open(CACHE / 'data' / 'lieder-d-v2' / 'index.jsonl')]
keep = train.reserved()
by_pdf = {}
for r in rows:
    m = re.search(r'p(\d+)-s(\d+)-0-1\.png$', r['image'])
    if m and r['part'] == 0 and r['staff'] == 1 and not train._is_reserved(r['pdf'], keep):
        by_pdf.setdefault(r['pdf'], []).append((int(m.group(2)), r))
limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
volumes = sorted(by_pdf)[:limit]
out = CACHE / 'data' / 'lieder-lyrics'
bar = Progress('lieder lyric lines', len(volumes), 'volumes', 'cutting the verses under every voice staff the Lieder pairing kept')
lines, staves = [], 0
for k, pdf in enumerate(volumes):
    bar.update(k, f'{pdf[:40]}: {len(lines)} lines so far')
    voice = []
    for system, r in by_pdf[pdf]:
        found = real._found(real.pages(real.SCANS / pdf)[r['page'] - 1])
        if system < len(found['systems']):
            voice.append({**r, 'at': found['systems'][system]['staves'][0]})
    staves += len(voice)
    lines += testsets.lyric_lines(voice, out, kind='lyrics-real')
with open(out / 'text' / 'lines.jsonl', 'w') as f:
    for line in lines:
        f.write(json.dumps(line, ensure_ascii=False) + '\n')
bar.finish(f'{len(lines)} lines from {staves} voice staves in {len(volumes)} volumes')
