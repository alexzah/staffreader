"""staffreader: optical music recognition, proved bar by bar.

  staffreader read <scan.pdf> --page N [-o page.musicxml]
      Read one page of a printed score (a PDF page, or a PNG) into
      MusicXML.
  staffreader fetch
      Download the trained models and the ladder's answer keys into the
      cache.
  staffreader suite [--level N] [--case ID ...] [--reader staffreader audiveris oemer] [--fresh]
      Read every case's pictures with each reader, score the readings
      against the answer keys, and write the scoreboard.
  staffreader layout [--level N] [--case ID ...]
      Find the staves, systems and barlines of every case's pages, score
      them against the right layout, and draw each page.
  staffreader compare <truth.musicxml> <reading.musicxml> [--bars A B]
      Score one reading against one answer.
  staffreader imslp <file-number> [--title WORK ...] [--search QUERY]
      Look up one IMSLP file: its edition, pages and copyright, and the
      link to download it by hand. Reads IMSLP's catalog only, within its
      rules (see imslp.py); never downloads.
  staffreader progress
      How far every long job has got: done of total, elapsed, time left.
  staffreader scans [--limit N]
      Fetch the scans OpenScore's quartets were transcribed from, out of
      the Internet Archive's IMSLP snapshot within its rules (archive.py),
      and list those to download by hand from IMSLP.

Improving the reader (evaluate, diagnose, testset, experiment) is
described in docs/improving.md.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import pathlib

from . import canonical, cases, layout_suite, metrics, render, report, suite


def _write_layout(image: pathlib.Path, page: int | None, json_path: pathlib.Path | None,
                  picture: pathlib.Path | None):
    """The page's layout as the reader found it, with each bar's box in
    reading order (bar 1 is the first system's first bar), so a bar of
    the reading can be cut from the straightened page beside it."""
    import cv2
    from . import layout
    gray = layout_suite.gray_page(image, page)
    lay = layout.read(gray)
    level = layout.level(gray, lay.angle, lay.bend)
    if picture:
        cv2.imwrite(str(picture), level)
    if json_path:
        d = lay.to_dict()
        bars = []
        for k, system in enumerate(lay.systems):
            staves = [lay.staves[i] for i in system.staves]
            left = min(st.x0 for st in staves)
            for right in system.barlines:
                bars.append({'system': k, 'left': round(left, 1), 'right': round(right, 1),
                             'top': round(staves[0].top, 1), 'bottom': round(staves[-1].bottom, 1)})
                left = right
        d['bars'] = bars
        d['space'] = round(lay.space, 2)
        json_path.write_text(json.dumps(d, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('suite')
    s.add_argument('--level', type=int)
    s.add_argument('--case', nargs='*')
    s.add_argument('--reader', nargs='*', default=['audiveris'])
    s.add_argument('--fresh', action='store_true', help='read again even where a reading is cached')
    s.add_argument('--out', type=pathlib.Path)
    lay = sub.add_parser('layout')
    lay.add_argument('--level', type=int)
    lay.add_argument('--case', nargs='*')
    lay.add_argument('--out', type=pathlib.Path)
    c = sub.add_parser('compare')
    c.add_argument('truth', type=pathlib.Path)
    c.add_argument('reading', type=pathlib.Path)
    c.add_argument('--bars', type=int, nargs=2)
    im = sub.add_parser('imslp')
    im.add_argument('file', nargs='?', help="the file's number, such as 04755; without one, every file on the work page")
    im.add_argument('--title', nargs='*', default=[], help='work pages to look on')
    im.add_argument('--search', help='a search for the work page, if the titles miss')
    pr = sub.add_parser('progress', help='how far every long job has got')
    pr.add_argument('--watch', type=float, nargs='?', const=10, metavar='SECONDS',
                    help='draw it again every few seconds (10 by default) until interrupted')
    sc = sub.add_parser('scans')
    sc.add_argument('--limit', type=int)
    ev = sub.add_parser('evaluate', help='measure a model on every test set and diagnose it (docs/improving.md)')
    ev.add_argument('model', type=pathlib.Path, help='a checkpoint, such as models/v2-mix/last.pt in the cache folder')
    ev.add_argument('--name', help='the evaluation folder, runs/eval-<name>; the model folder by default')
    ev.add_argument('--against', type=pathlib.Path, help='an earlier evaluation folder to compare with')
    ev.add_argument('--no-ladder', action='store_true')
    dg = sub.add_parser('diagnose', help='write the diagnosis page of an evaluation folder again')
    dg.add_argument('folder', type=pathlib.Path)
    dg.add_argument('--against', type=pathlib.Path)
    rd = sub.add_parser('read', help='read one page of a printed score into MusicXML')
    rd.add_argument('image', type=pathlib.Path, help='a PDF or a PNG')
    rd.add_argument('--page', type=int, help="the PDF's page, counted from 1")
    rd.add_argument('--pages', help="a run of the PDF's pages, such as 12-15, read as one work: its parts named on "
                                    "the first and matched on the rest by their labels")
    rd.add_argument('-o', '--out', type=pathlib.Path, help='where to write the MusicXML (default: beside the image)')
    rd.add_argument('--layout', type=pathlib.Path,
                    help="also write the page's staves, systems and bars as JSON, in the --picture's pixels")
    rd.add_argument('--picture', type=pathlib.Path, help='also write the straightened page the layout was found on')
    rd.add_argument('--names', type=pathlib.Path,
                    help="a JSON file naming each system's staves, top to bottom, as a list of lists (a list "
                         "of those for each page with --pages); used in place of the labels the reader reads")
    fe = sub.add_parser('fetch', help='download the trained models and the answer keys into the cache')
    fe.add_argument('--models', action='store_true', help='only the models')
    fe.add_argument('--answers', action='store_true', help='only the answer keys')
    st = sub.add_parser('stale', help='whether an evaluation folder still measures what one made now would')
    st.add_argument('folder', type=pathlib.Path)
    ts = sub.add_parser('testset', help='build the reserved test sets from the scans that never train')
    ts.add_argument('--model', type=pathlib.Path, help='the model that places systems; the default reader by default')
    ts.add_argument('--plan', action='store_true', help='only list the scans and their transcriptions')
    ts.add_argument('--only', nargs='*', help='only these kinds of music, such as orchestra')
    ex = sub.add_parser('experiment', help='train and compare runs as a written spec says (docs/improving.md)')
    ex.add_argument('spec', type=pathlib.Path)
    args = parser.parse_args()

    if args.command == 'evaluate':
        from . import evaluate
        model = args.model.expanduser()
        out = evaluate.evaluate(model, args.name or model.parent.name, args.against, ladder=not args.no_ladder)
        print(f'diagnosis at {out / "index.html"}')
        return

    if args.command == 'diagnose':
        from . import evaluate
        print(f'diagnosis at {evaluate.report(args.folder.expanduser(), args.against)}')
        return

    if args.command == 'read':
        import shutil
        import tempfile
        from .baselines import staffreader as ours
        image = args.image.expanduser().resolve()
        if image.suffix.lower() == '.pdf' and not (args.page or args.pages):
            raise SystemExit('name the page to read with --page, or a run of pages with --pages')
        if args.pages:
            first, last = (int(x) for x in args.pages.split('-'))
            pages = [(image, n) for n in range(first, last + 1)]
            label = f'-p{first}-{last}'
        else:
            pages = [(image, args.page if image.suffix.lower() == '.pdf' else None)]
            label = f'-p{args.page}' if args.page else ''
        out = args.out or image.with_name(f'{image.stem}{label}.musicxml')
        render.CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=render.CACHE) as work:
            names = json.loads(args.names.read_text()) if args.names else None
            if names is not None and not args.pages:
                names = [names]
            shutil.copy(ours.read_work(pages, pathlib.Path(work), names), out)
        if args.layout or args.picture:
            _write_layout(image, args.page if image.suffix.lower() == '.pdf' else None, args.layout, args.picture)
        print(out)
        return
    if args.command == 'fetch':
        from . import answers, models
        if not args.answers:
            for path in models.fetch():
                print(f'model in {path}')
        if not args.models:
            print(f'answer keys in {answers.fetch()}')
        return
    if args.command == 'testset':
        from . import testsets
        if args.plan:
            for pdf, p in sorted(testsets.plan().items()):
                print(f"{pdf[:60]:60} {p['music']:9} {'staff by staff' if p['staffwise'] else 'systems'}"
                      f"  {len(p['rows'])} transcriptions")
            return
        from .baselines import staffreader as ours
        print(testsets.build((args.model.expanduser() if args.model else ours.CHECKPOINT), args.only))
        return

    if args.command == 'experiment':
        from . import evaluate
        print(f'results at {evaluate.experiment(args.spec)}')
        return

    if args.command == 'stale':
        from . import evaluate
        why = evaluate.stale(args.folder.expanduser())
        print('\n'.join(f'stale: {w}' for w in why) if why else f'{args.folder.name} is current')
        raise SystemExit(1 if why else 0)

    if args.command == 'progress':
        from . import progress
        if args.watch is None:
            print(progress.report())
            return
        import time
        try:
            while True:
                print('\033[H\033[2J' + progress.report(), flush=True)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return

    if args.command == 'scans':
        from . import scans
        got = scans.bootstrap(args.limit)
        print(f"{len(got['fetched'])} fetched from the Archive; {len(got['flagged'])} to download by hand:")
        for r in got['flagged']:
            print(f"  {r['path']}  [{r['status']}]  {r['link']}")
        return

    if args.command == 'imslp':
        from . import imslp
        if args.file:
            found = imslp.find_file(args.file, args.title, args.search)
            print(json.dumps(found.to_dict() if found else None, indent=1, ensure_ascii=False))
            return
        # No file named: every file on the work page, to choose an edition from.
        titles = args.title or (imslp.search(args.search, 3)[:1] if args.search else [])
        for title in titles:
            pg = imslp.page(title)
            files = imslp.files_on(pg) if pg else []
            print(json.dumps({'work': title, 'files': [f.to_dict() for f in files]}, indent=1, ensure_ascii=False))
        return

    if args.command == 'compare':
        truth = canonical.read(args.truth)
        if args.bars:
            truth = truth.excerpt(*args.bars)
        print(json.dumps(metrics.compare(truth, canonical.read(args.reading)).summary(), indent=1))
        return

    if args.command == 'layout':
        out = args.out or render.CACHE / 'runs' / ('layout-' + datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))
        results = layout_suite.run(cases.all_cases(args.level, args.case), out)
        print(f'{len(results)} pages; layout scoreboard at {out / "index.html"}')
        return

    readers = [importlib.import_module(f'staffreader.baselines.{name}') for name in args.reader]
    out = args.out or render.CACHE / 'runs' / datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    results = suite.run(cases.all_cases(args.level, args.case), readers, out, args.fresh)
    page = report.write(out / 'results.json', out)
    print(f'{len(results)} runs; scoreboard at {page}')


if __name__ == '__main__':
    main()
