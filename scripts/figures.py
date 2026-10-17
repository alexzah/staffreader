"""The pictures in the README and docs/, made from a page staffreader has
read. Read the page first, keeping its layout and picture:

    uv run staffreader read score.pdf --page 1 -o page.musicxml --layout page.json --picture page.png

then:

    uv run python scripts/figures.py reading page.musicxml page.json page.png --system 1 --staves 8-11 \\
        --parts "violin 1" "violin 2" viola cello --edition "Mozart, Symphony No. 40, as Breitkopf & Härtel printed it" \\
        -o docs/images/reading.png
    uv run python scripts/figures.py layout page.json page.png -o docs/images/layout.png

`reading` sets a strip of the printed page above the same bars as read and
engraved by MuseScore; `layout` draws the staves, systems and barlines the
layout stage found over the page."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops, ImageDraw, ImageFont

from staffreader import render

WIDTH = 1800  # pixels across, for both pictures


def _trim(im: Image.Image, pad: int = 24) -> Image.Image:
    """[im] cut to its ink, with [pad] pixels of paper around it."""
    gray = im.convert('L')
    box = ImageChops.invert(gray).point(lambda v: 255 if v > 40 else 0).getbbox()
    if box is None:
        return im
    x0, y0, x1, y1 = box
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))


def _font(size: int, bold: bool) -> ImageFont.ImageFont:
    """Arimo, or Arial where Arimo is not installed."""
    for name in (('Arimo-Bold.ttf', 'Arial Bold.ttf') if bold else ('Arimo-Regular.ttf', 'Arial.ttf')):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def _labeled(im: Image.Image, label: str, note: str) -> Image.Image:
    """[im] under a band naming what it is: [label] in bold, [note] after it."""
    band = 64
    out = Image.new('RGB', (im.width, im.height + band), 'white')
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, im.width, band - 12], fill=(238, 241, 245))
    bold, plain = _font(34, True), _font(28, False)
    draw.text((20, (band - 12) // 2), label, font=bold, fill=(25, 35, 50), anchor='lm')
    x = 20 + draw.textlength(label, font=bold) + 18
    draw.text((x, (band - 12) // 2 + 2), note, font=plain, fill=(90, 100, 115), anchor='lm')
    out.paste(im, (0, band))
    return out


def _fit(im: Image.Image, width: int) -> Image.Image:
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


def _engraved(reading: pathlib.Path, first: int, last: int, parts: list[str], folder: pathlib.Path) -> Image.Image:
    """Bars [first] to [last] of the named [parts] of [reading], engraved
    by MuseScore on one wide system."""
    cut = render.excerpt(reading, first, last, folder / 'excerpt.musicxml')
    root = ET.parse(cut).getroot()
    names = {sp.get('id'): (sp.findtext('part-name') or '').strip() for sp in root.iter('score-part')}
    keep = {pid for pid, name in names.items() if name in parts}
    part_list = root.find('part-list')
    for sp in list(part_list):
        if sp.tag == 'score-part' and sp.get('id') not in keep:
            part_list.remove(sp)
    for part in list(root.findall('part')):
        if part.get('id') not in keep:
            root.remove(part)
    # A page wide enough for the bars on one system, and no title.
    for old in root.findall('defaults') + root.findall('work') + root.findall('movement-title'):
        root.remove(old)
    defaults = ET.Element('defaults')
    scaling = ET.SubElement(defaults, 'scaling')
    ET.SubElement(scaling, 'millimeters').text = '7'
    ET.SubElement(scaling, 'tenths').text = '40'
    layout = ET.SubElement(defaults, 'page-layout')
    ET.SubElement(layout, 'page-height').text = '900'
    ET.SubElement(layout, 'page-width').text = str(260 * (last - first + 1) + 400)
    root.insert(list(root).index(part_list), defaults)
    for part in root.findall('part'):
        for m in part.findall('measure'):
            for p in m.findall('print'):
                m.remove(p)
    xml = folder / 'strip.musicxml'
    ET.ElementTree(root).write(xml, encoding='utf-8', xml_declaration=True)
    out = folder / 'strip.png'
    subprocess.run([render.MSCORE, '-r', '300', '-o', str(out), str(xml)], env=render.mscore_env(),
                   capture_output=True, check=True)
    pages = sorted(folder.glob('strip*.png'))
    return _trim(Image.open(pages[0]).convert('RGB'))


def reading(musicxml: pathlib.Path, layout: pathlib.Path, picture: pathlib.Path, system: int,
            staves: tuple[int, int], parts: list[str], out: pathlib.Path,
            edition: str = 'the printed edition') -> pathlib.Path:
    """A strip of the printed page, [staves] (counted from 1, inclusive) of
    [system], above the same bars as staffreader read them."""
    lay = json.loads(layout.read_text())
    page = Image.open(picture).convert('RGB')
    sys_ = lay['systems'][system - 1]
    first_bar = sum(len(s['barlines']) for s in lay['systems'][:system - 1]) + 1
    bars = len(sys_['barlines'])
    top = lay['staves'][sys_['staves'][staves[0] - 1]]
    bottom = lay['staves'][sys_['staves'][staves[1] - 1]]
    space = lay['space']
    printed = page.crop((max(0, min(s['x0'] for s in (top, bottom)) - 22 * space), top['lines'][0] - 5 * space,
                         max(s['x1'] for s in (top, bottom)) + space, bottom['lines'][-1] + 5 * space))
    with tempfile.TemporaryDirectory() as tmp:
        engraved = _engraved(musicxml, first_bar, first_bar + bars - 1, parts, pathlib.Path(tmp))
    printed = _labeled(_fit(_trim(printed), WIDTH), 'Original', edition)
    engraved = _labeled(_fit(engraved, WIDTH), 'staffreader', 'digital score, engraved by MuseScore')
    gap = 36
    sheet = Image.new('RGB', (WIDTH, printed.height + gap + engraved.height), 'white')
    sheet.paste(printed, (0, 0))
    ImageDraw.Draw(sheet).line([(0, printed.height + gap // 2), (WIDTH, printed.height + gap // 2)],
                               fill=(200, 200, 200), width=2)
    sheet.paste(engraved, (0, printed.height + gap))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)
    return out


def layout(layout_json: pathlib.Path, picture: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """The page with each staff's lines in blue, each system boxed in green
    and each barline in red, as the layout stage found them."""
    lay = json.loads(layout_json.read_text())
    page = Image.open(picture).convert('RGB')
    page = Image.blend(page, Image.new('RGB', page.size, 'white'), 0.35)
    draw = ImageDraw.Draw(page)
    space = lay['space']
    for staff in lay['staves']:
        for y in staff['lines']:
            draw.line([(staff['x0'], y), (staff['x1'], y)], fill=(40, 110, 230), width=3)
    for s in lay['systems']:
        rows = [lay['staves'][i] for i in s['staves']]
        x0, x1 = min(r['x0'] for r in rows), max(r['x1'] for r in rows)
        y0, y1 = rows[0]['lines'][0], rows[-1]['lines'][-1]
        draw.rectangle([x0 - space, y0 - 2 * space, x1 + space, y1 + 2 * space], outline=(30, 160, 80), width=6)
        for x in s['barlines']:
            draw.line([(x, y0), (x, y1)], fill=(220, 50, 50), width=5)
    # The music only: the systems and the labels beside them.
    rows = lay['staves']
    box = (max(0, min(r['x0'] for r in rows) - 22 * space), rows[0]['lines'][0] - 5 * space,
           max(r['x1'] for r in rows) + 3 * space, rows[-1]['lines'][-1] + 2 * space)
    out.parent.mkdir(parents=True, exist_ok=True)
    _fit(page.crop(box), 1000).save(out, optimize=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('reading')
    r.add_argument('musicxml', type=pathlib.Path); r.add_argument('layout', type=pathlib.Path)
    r.add_argument('picture', type=pathlib.Path)
    r.add_argument('--system', type=int, default=1)
    r.add_argument('--staves', default='1-1', help='first and last staff of the system, from 1, such as 8-11')
    r.add_argument('--parts', nargs='+', required=True, help="the reading's part names to engrave")
    r.add_argument('--edition', default='the printed edition', help='what the original is, for its label')
    r.add_argument('-o', '--out', type=pathlib.Path, required=True)
    g = sub.add_parser('layout')
    g.add_argument('layout', type=pathlib.Path); g.add_argument('picture', type=pathlib.Path)
    g.add_argument('-o', '--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    if a.command == 'reading':
        first, last = (int(x) for x in a.staves.split('-'))
        print(reading(a.musicxml, a.layout, a.picture, a.system, (first, last), a.parts, a.out, a.edition))
    else:
        print(layout(a.layout, a.picture, a.out))
