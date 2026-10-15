"""Bars right on the Arabesque's real pages, split by whether the answer
holds a tuplet, and whether each wrong bar adds up.

E10's gate measured every staff holding a tuplet, most of them orchestral
with the "3" printed. The Arabesque's real pages print their triplets
without one, so they are the direct test. Each run is a suite run of the
five Arabesque cases with one model:

  STAFFREADER_MODEL=<checkpoint> uv run staffreader suite --reader staffreader \\
      --case arabesque-1-opening arabesque-1-p2 arabesque-1-p3 arabesque-1-p4 arabesque-1-p5 \\
      --out <cache>/runs/<name>
  uv run python experiments/ladder_tuplets.py <name> [<name> ...]

E10 (2026-10-15): control 27 of 78 triplet bars right, variant 31; of the
variant's wrong triplet bars 30 of 47 do not add up, of its other wrong
bars 0 of 7. A bar that does not add up marks a missed triplet."""

import json
import sys
import xml.etree.ElementTree as ET

from staffreader.render import CACHE


def tuplet_bars() -> set[str]:
    root = ET.parse(CACHE / 'answers' / 'arabesque-1.musicxml').getroot()
    return {m.get('number') for part in root.iter('part') for m in part.iter('measure')
            if m.find('.//time-modification') is not None}


def main():
    tuplets = tuplet_bars()
    for name in sys.argv[1:]:
        counts = {'tuplets': [0, 0, 0], 'none': [0, 0, 0]}
        for r in json.load(open(CACHE / 'runs' / name / 'results.json')):
            if not r['track'].startswith('scan'):
                continue
            for b in r['bars']:
                if b['gt'] is None or b['pred'] is None:
                    continue
                c = counts['tuplets' if b['number'] in tuplets else 'none']
                c[0 if b['exact_full'] else 1 if b['valid'] else 2] += 1
        for kind, (right, adds, short) in counts.items():
            print(f'{name}  {kind:8}  right {right:3}  wrong and adds up {adds:3}  wrong and does not add up {short:3}')


if __name__ == '__main__':
    main()
