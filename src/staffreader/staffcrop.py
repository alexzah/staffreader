"""One staff cut out of a page, the same way for training and for reading.

The cut is centered on the staff's middle line and is SPACES staff
spaces tall, scaled so a staff space is PX_PER_SPACE pixels. It runs from
a little before the staff's left end to a little after its right end, so
the clef and the closing barline are whole. Neighboring staves, lyrics and
dynamics that fall inside the window stay in it, as they do on a page."""

from __future__ import annotations

import cv2
import numpy as np

SPACES = 18
PX_PER_SPACE = 16
HEIGHT = SPACES * PX_PER_SPACE
MAX_WIDTH = 4096


def crop(gray: np.ndarray, lines: list[float], x0: float, x1: float, *,
         spaces: float = SPACES, px_per_space: float = PX_PER_SPACE) -> np.ndarray:
    """The staff whose five lines lie at [lines] (pixel heights, top to
    bottom) and which runs from [x0] to [x1], as a gray image."""
    space = (lines[-1] - lines[0]) / (len(lines) - 1)
    center = (lines[0] + lines[-1]) / 2
    zoom = px_per_space / space
    left = x0 - 1.5 * space
    width = min(MAX_WIDTH, int(round((x1 + 1.0 * space - left) * zoom)))
    height = int(round(spaces * px_per_space))
    # Map the window onto the output with one affine warp: subpixel exact,
    # and anything outside the page is paper.
    top = center - spaces * space / 2
    m = np.array([[zoom, 0, -left * zoom], [0, zoom, -top * zoom]], np.float32)
    interp = cv2.INTER_AREA if zoom < 1 else cv2.INTER_LINEAR
    return cv2.warpAffine(gray, m, (width, height), flags=interp, borderValue=255)
