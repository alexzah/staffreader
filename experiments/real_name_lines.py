"""Real staff labels for the text reader, with their printed text.

Pairing says which instrument each staff of a paired system is, but the
transcriptions name parts in their own words (OpenScore's orchestral ones
in standard English: 'Bsn 1'), not as the page prints them ('Fg.'). So
each label is found left of its system as the reader finds it
(baselines.staffreader._margin_labels), read by the text reader, and given
the printed form, among those instruments.HISTORICAL holds for the
staff's known instrument (with its number and key), closest to what was
read, where it is close: 'olin I.' on a first violin's staff becomes
'Violino I.'. A label read far from every form of its instrument, or one
of more than one line, is left out. The lines go to DATA/real-names/text.

  uv run python experiments/real_name_lines.py [set ...]"""

import difflib
import json
import re
import sys
from types import SimpleNamespace

import cv2

from staffreader import instruments, layout, lyrics, text
from staffreader.baselines import staffreader as ours
from staffreader.progress import Progress
from staffreader.real import SCANS, _found, pages
from staffreader.render import CACHE
from relabel_v3 import source
from staffreader.data import _parts, _root

SETS = sys.argv[1:] or ['orch-c-v2', 'real3-v2', 'pdmx-a3', 'pdmx-b-ens']
OUT = CACHE / 'data' / 'real-names'
PLACE = re.compile(r'p(\d+)-s(\d+)-n(\d+)\.png$')
NUMBERS = ['', ' {r}', ' {r}.', ' {n}', ' {n}.', '{n}. ']
ROMAN = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}


def forms(inst: instruments.Instrument) -> list[str]:
    """Every way the table prints this instrument, with its number and,
    for one that transposes, a key."""
    fulls, shorts = instruments.HISTORICAL[inst.name]
    keys = [''] + ([f' in {k.upper() if len(k) == 1 else k.capitalize()}' for k in instruments.SOUNDING[inst.name]]
                   + [f' ({k.upper() if len(k) == 1 else k.capitalize()})' for k in instruments.SOUNDING[inst.name]]
                   if inst.name in instruments.SOUNDING else [])
    out = []
    for base in fulls + shorts:
        for number in (NUMBERS if inst.number in ROMAN else ['']):
            n = number.format(r=ROMAN.get(inst.number, ''), n=inst.number or '')
            for key in keys:
                out.append(f'{n}{base}{key}' if n.endswith('. ') else f'{base}{n}{key}')
    return out


def closest(read: str, inst: instruments.Instrument) -> str | None:
    """The printed form nearest what was read, where it is near."""
    plain = ' '.join(read.lower().split())
    scored = sorted(((difflib.SequenceMatcher(None, plain, f.lower()).ratio(), f) for f in forms(inst)), reverse=True)
    return scored[0][1] if scored and scored[0][0] >= 0.6 else None


def main():
    net = lyrics.text_reader()
    device = next(net.parameters()).device
    names_of: dict[str, list[str]] = {}
    systems: dict[tuple, dict[int, instruments.Instrument]] = {}  # (set, pdf, page, system) -> staff -> instrument
    for s in SETS:
        folder = CACHE / 'data' / s
        for line in open(folder / 'index.jsonl'):
            r = json.loads(line)
            place = PLACE.search(r['image'])
            if not place:
                continue
            key = r['mxl']
            if key not in names_of:
                try:
                    root = _root(source(r, folder))
                    names_of[key] = [sp.findtext('part-name') or '' for sp in root.iter('score-part')]
                except Exception:
                    names_of[key] = []
            names = names_of[key]
            if r['part'] >= len(names):
                continue
            inst = instruments.instrument(names[r['part']])
            if inst is None or inst.name not in instruments.HISTORICAL:
                continue
            if r.get('with') is not None:  # two players on one staff: the label names the instrument alone
                inst = instruments.Instrument(inst.name, None, inst.transpose)
            j, k, n = (int(x) for x in place.groups())
            systems.setdefault((s, r['pdf'], j, k), {})[n] = inst
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'text').mkdir(exist_ok=True)
    bar = Progress('real name lines', len(systems), 'systems', 'staff labels on paired pages, given their printed form')
    kept = left = 0
    pdfs = {}
    with open(OUT / 'text' / 'lines.jsonl', 'w') as out:
        for i, ((s, pdf, j, k), staves) in enumerate(sorted(systems.items())):
            if pdf not in pdfs:
                scan = next(iter(sorted(SCANS.glob(pdf))), None)
                pdfs[pdf] = pages(scan) if scan else None
            pngs = pdfs[pdf]
            if not pngs or j - 1 >= len(pngs):
                continue
            found = _found(pngs[j - 1])
            gray = cv2.imread(str(pngs[j - 1]), cv2.IMREAD_GRAYSCALE)
            level = layout.level(gray, found['angle'], found.get('bend'))
            staff_objs = [SimpleNamespace(lines=st['lines'], top=st['lines'][0], bottom=st['lines'][4], x0=st['x0'],
                                          x1=st['x1'], space=(st['lines'][4] - st['lines'][0]) / 4)
                          for st in found['staves']]
            system = SimpleNamespace(staves=found['systems'][k]['staves'], parts=None)
            lay = SimpleNamespace(staves=staff_objs)
            groups = [[st] for st in system.staves]
            labels = ours._margin_labels(level, lay, system, groups)
            one_line = [(cov, lines[0]) for cov, lines in labels if len(lines) == 1]
            if one_line:
                x, w = text.batch([text.view(c) for _, c in one_line], device)
                for (cov, crop), read in zip(one_line, net.read(x, w)):
                    insts = {staves.get(g) for g in cov}
                    if None in insts or len({t.name for t in insts}) != 1:
                        left += 1
                        continue
                    inst = next(iter(insts)) if len(insts) == 1 else instruments.Instrument(next(iter(insts)).name)
                    printed = re.findall(r'\b(I{1,3}|IV|[1-4])\b', read.replace('.', ' '))
                    if printed and inst.number and ROMAN.get(inst.number) not in printed and inst.number not in printed:
                        left += 1  # the print numbers it otherwise than the transcription: trust neither
                        continue
                    said = instruments.instrument(read)
                    if said is not None and said.name != inst.name:
                        left += 1  # the label names another instrument than pairing found: trust neither
                        continue
                    # Only a label read exactly as a printed form of its instrument is kept: forcing a
                    # misread onto the nearest form was wrong too often by eye ('Dbn.', a double bassoon,
                    # made 'Db.'; 'Hp. II' made 'Hp.'), so these lines teach the real typefaces of labels
                    # the reader reads right, and the engraved historical labels teach the forms.
                    target = read.strip() if read.strip() in forms(inst) else None
                    if target is None:
                        left += 1
                        continue
                    name = f"{pdf.split('-')[0]}-p{j:04d}-s{k}-g{cov[0]}.png"
                    cv2.imwrite(str(OUT / 'text' / name), crop)
                    out.write(json.dumps({'image': f'text/{name}', 'text': target, 'kind': 'name-real', 'read': read},
                                         ensure_ascii=False) + '\n')
                    kept += 1
            if i % 50 == 0:
                bar.update(i, f'{kept} labels kept, {left} left out')
    bar.finish(f'{kept} labels kept, {left} left out')


if __name__ == '__main__':
    main()
