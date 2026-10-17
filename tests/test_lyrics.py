import cv2
import numpy as np

from staffreader import layout, lyrics


def _page(verses: int, stems: bool) -> tuple[np.ndarray, layout.Staff]:
    """A staff 20 px a space, verses of words under it, and optionally
    stems and a beam hanging below the staff."""
    page = np.full((600, 1200), 255, np.uint8)
    lines = [100 + 20 * k for k in range(5)]
    for y in lines:
        page[y:y + 2, 50:1150] = 0
    if stems:
        for x in range(100, 1100, 60):
            page[180:215, x:x + 2] = 0
        for x in range(100, 1100, 250):
            page[209:215, x:x + 182] = 0
    for v in range(verses):
        y = 250 + 50 * v
        for x in range(80, 1000, 180):
            cv2.putText(page, 'Lie- be', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 2)
    return page, layout.Staff([float(y) for y in lines], 50.0, 1150.0)


def test_each_verse_is_one_row():
    page, staff = _page(3, stems=False)
    assert len(lyrics.rows(page, staff, 600)) == 3


def test_stems_below_the_staff_are_not_a_verse():
    page, staff = _page(3, stems=True)
    found = lyrics.rows(page, staff, 600)
    assert len(found) == 3 and found[0][0] > 216


def test_a_slur_from_the_staff_below_does_not_swallow_the_last_verse():
    page, staff = _page(3, stems=False)
    cv2.ellipse(page, (800, 420), (300, 90), 0, 180, 360, 0, 3)  # arcs up past the last verse
    assert len(lyrics.rows(page, staff, 440)) == 3


def test_a_hyphen_the_reader_missed_is_put_back_where_the_language_divides_the_word():
    # Brahms, Regenlied: 'at-mend' and 'Her-zens' are printed tight and read whole.
    line = 'stand die See- le atmend of- fen, an des Herzens Klop- fen,'
    assert lyrics.mend(line, 2) == 'stand die See- le at- mend of- fen, an des Her- zens Klop- fen,'


def test_a_verse_with_a_syllable_for_every_note_is_left_as_read():
    assert lyrics.mend('Schauernd kühl- te', 0) == 'Schauernd kühl- te'
