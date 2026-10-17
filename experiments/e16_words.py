"""E16: does the text reader learn directions in words from engraved lines?

The staff reader spells directions in words (dolce, a 2., Allegro vivace)
a letter at a time and gets them right in 38% of cases on real pages and
far fewer on engraved ones; on the Wedding March's first page it made up
"II. Allegro molto" and "arco". The text reader reads 93% of engraved
lyric and name lines exactly but has never seen a direction. Proposal:
engraved direction lines (data/r3-words, cut by data._text_lines from the
engraver's StaffText and Tempo boxes) teach it to, and the reader can then
read directions as it reads staff labels.

Two runs from text-v3 for the same steps and rate: the control on
text-v3's own lines, the variant with the direction lines too. Each is
read on held-out direction lines, held-out engraved lyric and name lines,
and the reserved real lyric lines. The variant passes if direction lines
get better beyond a paired bootstrap's 90% interval and neither of the
others gets worse beyond theirs. Results: runs/e16-words.json."""

import json
import random

from e9_lyrics import paired, read_all, score
from staffreader import text
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

BASE = CACHE / 'models' / 'text-v3' / 'last.pt'
RECIPE = ['r2-single', 'r2-grand', 'r2-tenor', 'lieder-lyrics', 'r3-names', 'real-names']
WEIGHTS = {'lyrics-real': 0.4, 'name-historical': 0.6, 'name-real': 0.3}
RUNS = {'e16-control': RECIPE, 'e16-words': RECIPE + ['r3-words-only']}
STEPS = 3000
LR = 1e-4


def main():
    bar = Progress('e16 words', len(RUNS) + 1, 'stages', 'text reader: its own lines against them and direction lines')
    parent_of_children('e16 words')
    for k, (name, dirs) in enumerate(RUNS.items()):
        bar.update(k, f'training {name}')
        if not (CACHE / 'models' / name / 'last.pt').exists():
            text.train(name, dirs, steps=STEPS, lr=LR, resume=str(BASE), weights=WEIGHTS)
    bar.update(len(RUNS), 'reading the test lines')
    rng = random.Random(1)
    words = text.lines(['r3-words-only'], True)
    sets = {'words': rng.sample(words, min(1000, len(words))),
            'rendered': rng.sample([r for r in text.lines(RECIPE[:3], True)], 1000),
            'real': text.lines(['test-lyrics'], None)}
    found = {name: {kind: read_all(CACHE / 'models' / name / 'last.pt', rows) for kind, rows in sets.items()}
             for name in RUNS}
    summary = {name: {kind: score(rows) for kind, rows in d.items()} for name, d in found.items()}
    c, v = found['e16-control'], found['e16-words']
    changes = {kind: paired(c[kind], v[kind], 'wrong', 'chars') for kind in sets}
    better = changes['words']['interval'][1] < 0
    worse = [k for k in ('rendered', 'real') if changes[k]['interval'][0] > 0]
    verdict = 'passes' if better and not worse else (f'fails: {", ".join(worse)} worse' if better
                                                      else 'fails: direction lines not better')
    examples = [{'truth': a['truth'], 'control': a['read'], 'words': b['read']}
                for a, b in list(zip(c['words'], v['words']))[:20]]
    (CACHE / 'runs' / 'e16-words.json').write_text(json.dumps(
        {'summary': summary, 'changes': changes, 'verdict': verdict, 'examples': examples}, indent=1, ensure_ascii=False))
    s = summary
    bar.finish(f"{verdict}: direction lines exact {100 * s['e16-control']['words']['exact']:.1f}% to "
               f"{100 * s['e16-words']['words']['exact']:.1f}%")


if __name__ == '__main__':
    main()
