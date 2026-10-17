"""The word finder (words.py): teach it on engraved pages, then measure its
boxes on held-out engraved pages against the engraver's own.

  uv run python experiments/finder.py train <name> [steps]
  uv run python experiments/finder.py measure <name>"""

import sys

import cv2
import numpy as np

from staffreader import words
from staffreader.render import CACHE

SETS = ['r3-wordpages']


def measure(name: str) -> None:
    net = words.load(CACHE / 'models' / name / 'last.pt')
    tp = fp = fn = 0
    for base, r in words.rows(SETS, True)[:400]:
        gray = cv2.imread(str(base / r['image']), cv2.IMREAD_GRAYSCALE)
        found = words.find(net, gray, r['space'])
        true = [tuple(b) for b in r['words']]
        hit = set()
        for f in found:
            best = max(range(len(true)), key=lambda k: _iou(f, true[k]), default=None)
            if best is not None and _iou(f, true[best]) >= 0.3 and best not in hit:
                hit.add(best)
                tp += 1
            else:
                fp += 1
        fn += len(true) - len(hit)
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    print(f'{name}: words found {tp} of {tp + fn} (recall {r:.3f}), precision {p:.3f}')


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        words.train(sys.argv[2], SETS, steps=int(sys.argv[3]) if len(sys.argv) > 3 else 6000)
    else:
        measure(sys.argv[2])
