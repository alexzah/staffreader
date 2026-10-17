# How it is measured

Every claim about how well staffreader reads is a number measured on pages whose right answer is known. There are two kinds of test:

- **The ladder**: a few dozen pages, each a case written by hand, climbing from one line on one staff to an orchestral page. Most are pages of the very printed edition their answer key was transcribed from, so there is no edition difference to argue about.
- **The reserved test sets**: thousands of real staves from volumes that never train, built by `staffreader testset` and measured by `staffreader evaluate` ([improving.md](improving.md)).

## Running the ladder

```bash
uv run staffreader fetch --answers                             # the answer keys
uv run staffreader suite --reader staffreader audiveris        # every case, every reader, and the scoreboard
uv run staffreader suite --case paganini-24-theme
uv run staffreader layout                                      # staves, systems and barlines on every page
uv run staffreader compare truth.musicxml reading.musicxml --bars 1 16
```

Each run writes `runs/<time>/index.html` in the cache: a scoreboard by case and reader, with each wrong bar shown as the page printed it, the answer and the reading. A reading is cached by the picture it was read from, and `--fresh` reads everything again.

## Cases

`cases/*.yaml` names an answer key, the bars, and the pictures:

- **synthetic**: the bars engraved by MuseScore in a font the reader never trains on, clean or made to look scanned, whose answer is exact;
- **scan**: a page of a printed edition, the one its answer key was transcribed from, with each system's staves and bars counted by eye.

The level runs from 1, one line on one staff, up to 6, orchestra and voices. A case may leave unscored a family of marks its answer key does not hold as printed (`masks`), or name parts its edition prints on another part's staff (`shared`), such as a cello-and-bass staff the transcription writes as two parts.

## Answer keys

`answers/keys.yaml` lists every answer key: its source, its license, and any corrections, each read from the printed page. All are CC0 or public domain. They are published as one zip on the `answers-v1` release, never in git, and `staffreader fetch` downloads it and checks its checksum. `scripts/build_answer_keys.py` makes the zip.

## Scans

The printed scans come from IMSLP, which allows downloading only by hand. `staffreader imslp <file-number>` prints a file's edition, pages and copyright, and the link to download it into your scans folder (the `scans` setting). A case names its scan by IMSLP's file name, and a case whose scan is missing is skipped.

## How a reading is scored

`canonical.py` reads any MusicXML into events per measure: notes, rests and grace notes. Each event has its onset in the bar as an exact fraction, its written pitch, its length, its ties and its staff. Voices are ignored, and anything the score hides is left out. Each measure also carries the clefs, key and time in force, what it prints, its repeats and endings, its lyrics, and every mark the reader is meant to read.

`metrics.py` pairs the reading's parts with the score's by their music. It lines up each part's measures with Needleman–Wunsch, so a missed barline costs one bar and not every bar after it. It then scores:

- notes at three levels: **Core** (pitch and onset), **Rhythm** (and length) and **Full** (and ties and staff);
- rests;
- **bars right**, the share of bars needing no correction, which is the headline;
- read bars that add up, which needs no answer and so also serves as a confidence signal;
- the clefs, key and time in force, and the repeats and endings;
- each family of marks, and lyrics;
- each mistake, by kind.

The metric is tested on itself: an answer key scores 100% against itself and against MuseScore's export of it, and each kind of mistake Audiveris made on a real page, injected into a known score, is charged exactly once (`tests/test_metrics.py`).

`layout_truth.py` gives the right layout of a page. For an engraved page it comes from MuseScore's SVG, which marks every staff line and barline; for a scan it is the staves and bars counted in the case. `staffreader layout` scores every page and draws the found layout over it.

## The other readers

A reader is a module in `staffreader/baselines` with a `NAME` and `read(image, page, work)`, which returns a MusicXML file. Besides staffreader itself:

- **audiveris** runs Audiveris 5.11: the page rendered so its staff lines lie about 20 pixels apart, read in batch with Italian text. It runs the Flathub build unless the `audiveris` setting names another command.
- **oemer** (MIT) runs from its own environment, `venvs/oemer` in the cache, with Python 3.10, `onnxruntime==1.17.3` (CPU) and `numpy==1.23.5`. Newer onnxruntime refuses its model.
