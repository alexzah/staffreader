"""E9: does the text reader learn real printed lyrics from real lines?

The text reader reads 92% of rendered lines exactly and 1% of real lyric
lines (eval-v2-mix); even on letters alone, ignoring spaces, hyphens and
verse numbers, 28% of a real line's letters are wrong. It has only seen
MuseScore's fonts and spacing. Proposal: the cause is that domain gap, so
real lines from the Lieder volumes that train (data/lieder-lyrics, 6,469
lines, kind lyrics-real, a third of batches) should close much of it
without costing rendered lines.

Two runs from text-a for the same steps and rate: the control on the
rendered lines alone, the variant with the real lines too. Each is read
line by line on the reserved real lines (data/test-lyrics, volumes that
never train) and on held-out rendered lines; the variant passes if real
lines get better beyond a paired bootstrap's 90% interval and rendered
lines do not get worse beyond theirs. Results: runs/e9-lyrics.json."""

import json
import random
import sys
import re
import difflib

import cv2
import torch

from staffreader import text
from staffreader.progress import Progress, parent_of_children
from staffreader.render import CACHE

BASE = CACHE / 'models' / 'text-a' / 'last.pt'
RENDERED = ['r2-single', 'r2-grand', 'r2-tenor']
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
TAG = sys.argv[2] if len(sys.argv) > 2 else ''  # a pilot's runs are named apart
LR = 1e-4
RUNS = {f'e9{TAG}-control': RENDERED, f'e9{TAG}-real': RENDERED + ['lieder-lyrics']}
# E9b: the real lines a sixth of batches rather than a third, to see whether the rendered lines'
# small loss in E9 came from their smaller share of batches.
WEIGHTS = {}
if TAG == 'b':
    RUNS = {'e9-control': RENDERED, 'e9b-real': RENDERED + ['lieder-lyrics']}
    WEIGHTS = {'e9b-real': {'lyrics-real': 0.4}}


def letters(s: str) -> str:
    s = re.sub(r'^\s*\d+\.\s*', '', s).replace('ß', 'ss')
    return re.sub(r'[\s\-_]', '', s)


def wrong(a: str, b: str) -> int:
    sm = difflib.SequenceMatcher(None, a, b)
    return max(len(a), len(b)) - sum(m.size for m in sm.get_matching_blocks())


def read_all(model, items) -> list[dict]:
    net = text.load(model)
    out = []
    for k in range(0, len(items), 16):
        rows = items[k:k + 16]
        x, w = text.batch([text.view(cv2.imread(r['path'], cv2.IMREAD_GRAYSCALE)) for r in rows],
                          next(net.parameters()).device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            got = net.read(x, w)
        for r, g in zip(rows, got):
            out.append({'truth': r['text'], 'read': g, 'exact': g == r['text'], 'wrong': wrong(r['text'], g),
                        'chars': len(r['text']), 'letters_wrong': wrong(letters(r['text']), letters(g)),
                        'letters': len(letters(r['text']))})
    return out


def score(lines: list[dict]) -> dict:
    n = max(1, len(lines))
    return {'lines': len(lines), 'exact': sum(l['exact'] for l in lines) / n,
            'cer': sum(l['wrong'] for l in lines) / max(1, sum(l['chars'] for l in lines)),
            'letters_cer': sum(l['letters_wrong'] for l in lines) / max(1, sum(l['letters'] for l in lines))}


def paired(before: list[dict], after: list[dict], field: str, total: str, rounds: int = 2000) -> dict:
    """The change in a rate (after minus before) with a paired bootstrap
    90% interval over lines."""
    rng = random.Random(0)
    n = len(before)
    def rate(rows, idx):
        return sum(rows[i][field] for i in idx) / max(1, sum(rows[i][total] for i in idx))
    base = list(range(n))
    change = rate(after, base) - rate(before, base)
    draws = sorted(rate(after, idx) - rate(before, idx)
                   for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(rounds)))
    return {'change': change, 'interval': [draws[int(0.05 * rounds)], draws[int(0.95 * rounds)]]}


def main():
    bar = Progress('e9 lyrics', len(RUNS) + 1, 'stages', 'text reader: rendered lines alone against rendered and real lines')
    parent_of_children('e9 lyrics')
    for k, (name, dirs) in enumerate(RUNS.items()):
        bar.update(k, f'training {name}')
        if not (CACHE / 'models' / name / 'last.pt').exists():
            text.train(name, dirs, steps=STEPS, lr=LR, resume=str(BASE), weights=WEIGHTS.get(name))
    bar.update(len(RUNS), 'reading the test lines')
    real = text.lines(['test-lyrics'], None)
    rendered = random.Random(1).sample(text.lines(RENDERED, True), 1000)
    found = {}
    for name in ['text-a', *RUNS]:
        model = BASE if name == 'text-a' else CACHE / 'models' / name / 'last.pt'
        found[name] = {'real': read_all(model, real), 'rendered': read_all(model, rendered)}
    summary = {name: {kind: score(rows) for kind, rows in d.items()} for name, d in found.items()}
    control, variant = list(RUNS)
    c, v = found[control], found[variant]
    changes = {kind: {'cer': paired(c[kind], v[kind], 'wrong', 'chars'),
                      'letters_cer': paired(c[kind], v[kind], 'letters_wrong', 'letters')} for kind in ('real', 'rendered')}
    better = changes['real']['letters_cer']['interval'][1] < 0
    no_worse = changes['rendered']['cer']['interval'][0] <= 0 or changes['rendered']['cer']['change'] <= 0
    verdict = 'passes' if better and no_worse else ('fails: rendered lines worse' if better else 'fails: real lines not better')
    examples = [{'truth': a['truth'], 'control': a['read'], 'real': b['read']}
                for a, b in list(zip(c['real'], v['real']))[:12]]
    (CACHE / 'runs' / f'e9{TAG}-lyrics.json').write_text(json.dumps(
        {'summary': summary, 'changes': changes, 'verdict': verdict, 'examples': examples}, indent=1, ensure_ascii=False))
    bar.finish(f"{verdict}: real lines {100 * summary[control]['real']['letters_cer']:.1f}% "
               f"to {100 * summary[variant]['real']['letters_cer']:.1f}% of letters wrong; rendered "
               f"{100 * summary[control]['rendered']['cer']:.2f}% to {100 * summary[variant]['rendered']['cer']:.2f}%")


if __name__ == '__main__':
    main()
