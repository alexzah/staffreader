# staffreader

Optical music recognition trained only on free data, proved bar by bar against scores whose answer is known.

Much printed music is in the public domain but has no free digital edition. staffreader reads a page of a printed score into MusicXML. It is measured on a ladder of real printed pages whose exact transcription exists, so every claim about how well it reads is a number you can check.

- **Free data only.** It is trained on PDMX's CC0 and public-domain scores, engraved by MuseScore, and on public-domain printed scans paired with OpenScore's CC0 transcriptions. Nothing non-commercial was trained on, so the model carries no conditions beyond its own license.
- **Measured on real pages.** The test pages are printed editions whose answer keys were transcribed from them or checked against them, bar by bar: string quartets, Lieder, solo piano and orchestral scores. Works that are tested on never train.
- **Read staff by staff.** A classical layout stage finds the staves, systems and barlines. Then a ConvNeXt encoder and transformer decoder of 26M parameters reads each staff's notes and marks, and a small text reader reads lyrics and instrument names.

## How well it reads

Real printed pages, as of October 2026, with the model `m4b-mix` (notes right at pitch and onset, as F1, and the share of bars needing no correction), against Audiveris 5.11:

| Pages | staffreader | Audiveris |
|---|---|---|
| String quartets (4 pages: Lark, Dissonance, Emperor, Boccherini) | 0.97 notes, 91% of bars | 0.38–0.77 notes, 23–31% of bars; no reading of two pages |
| Voice and piano (4 Lieder pages) | 0.99 notes, 89% of bars | 0.80 notes, 53% of bars |
| Orchestra (Mozart's 40th, 6 pages) | 0.97 notes, 92% of bars | 0.89 notes, 69% of bars |
| Solo piano (Debussy's first Arabesque, 5 pages) | 0.85 notes, 47% of bars | not measured |

Solo piano is the weakest kind of music today. A model that also reads every mark of the inventory (articulations, dynamics, slurs, ornaments, fingering and more) and lyrics is in training.

## Installing

You need [uv](https://docs.astral.sh/uv/), Python 3.13 (uv fetches it), and poppler's `pdftoppm`. An NVIDIA GPU makes reading fast; the CPU works, slowly. Rendering training data and the engraved test pages needs MuseScore 4.

```bash
git clone https://github.com/alexzah/staffreader && cd staffreader
uv sync --group dev
uv run pytest
```

The trained model is published with the first release.

## Reading a page

```bash
uv run staffreader read edition.pdf --page 12 -o page12.musicxml
uv run staffreader read page.png
```

A page is read on its own; joining the pages of a work into one score is still to come.

## Settings

staffreader keeps its work (renders, data sets, models and runs) in `~/.cache/staffreader`, never in git. Everything it needs to know about your machine goes in `~/.config/staffreader/settings.toml`, and any setting can be given instead as an environment variable, `STAFFREADER_<NAME>`:

```toml
cache = "~/.cache/staffreader"   # renders, data sets, models and runs
scans = "~/scores/imslp"         # printed scans, named as IMSLP names them
pdmx = "~/datasets/pdmx"         # the PDMX dataset, for rendering training data
contact = "you@example.org"      # sent with every request to IMSLP, the Internet Archive and the MDZ
mscore = "mscore"                # MuseScore 4
device = "cuda"                  # or "cpu"
```

`settings.py` lists every setting.

## The test ladder

```bash
uv run staffreader fetch                                       # the answer keys
uv run staffreader suite --reader staffreader audiveris         # every case, every reader, and the scoreboard
uv run staffreader suite --case paganini-24-theme
uv run staffreader layout                                       # staves, systems and barlines on every page
uv run staffreader compare truth.musicxml reading.musicxml --bars 1 16
```

Each run writes `runs/<time>/index.html` in the cache. A reading is cached by the picture it was read from, and a reader that failed on a picture is not run on it again, so `--fresh` reads everything again.

### Cases

`cases/*.yaml` names an answer key, the bars, and the pictures:

- **synthetic**: the bars engraved by MuseScore, clean or made to look scanned, whose answer is exact;
- **scan**: a page of a printed edition, the one its answer key was transcribed from, with each system's staves and bars counted by eye.

The level runs from 1, one line on one staff, up to 6, orchestra and chorus.

### Answer keys

`answers/keys.yaml` lists every answer key: its source, its license, and any corrections, each read from the printed page. All are CC0 or public domain. They are published as one zip on a GitHub release, never in git, and `staffreader fetch` downloads it and checks its checksum.

### Scans

The printed scans come from IMSLP, which allows downloading only by hand. `staffreader imslp <file-number>` prints a file's edition, pages and copyright, and the link to download it into your scans folder. A case names its scan by IMSLP's file name.

## How a reading is scored

`canonical.py` reads any MusicXML into events per measure: notes, rests and grace notes. Each event has its onset in the bar as an exact fraction, its written pitch, its length, its ties and its staff. Voices are ignored. Anything the score hides is left out. Each measure also carries the clefs, key and time in force, what it prints, its repeats and endings, its lyrics, and every mark of the inventory.

`metrics.py` pairs the reading's parts with the score's by their music. It lines up each part's measures with Needleman–Wunsch, so a missed barline costs one bar and not every bar after it. It then scores:

- notes at three levels: **Core** (pitch and onset), **Rhythm** (and length) and **Full** (and ties and staff);
- rests;
- **bars right**, the share of bars needing no correction, which is the headline;
- read bars that add up, which needs no answer and so also serves as a confidence signal;
- the clefs, key and time in force, and the repeats and endings;
- each family of marks, and lyrics;
- each mistake, by kind.

A case may leave a family of marks unscored where its answer key does not hold it as printed (`masks`). The tests in `tests/` inject each error Audiveris made on the Ricordi Boccherini page into a known score, and check the metric charges exactly that error.

## The layout stage

`layout.py` finds a page's staves, groups them into systems and finds each system's barlines, with no learning:

1. **Straightening.** The paper is evened out against a blurred copy of itself and thresholded. Anything clearly darker than the paper counts as ink, so faint lines survive. The page is cut into vertical strips, and the shift of the staff lines from strip to strip gives the tilt to undo. A page that bends is flattened, band by band.
2. **Line thickness and spacing.** These are the most common vertical runs of ink and of paper.
3. **Staves.** Staff lines are what survives a long horizontal opening, after short gaps are bridged. Every run of five lines a staff space apart is a candidate, and candidates are taken longest first. Each of the five lines must run most of the staff's length, so a ledger line never stands in for a staff line. A staff's ends are followed out past clefs and key signatures, tracking each line as it drifts.
4. **Systems.** Staves joined at their left edge by a line are one system.
5. **Barlines.** A barline fills a staff from top line to bottom line. A stroke with a notehead touching it is a stem, and so is one that reaches past the staff; repeat dots stand apart and do not count. Most of a system's staves must agree on a barline. Where a system's barlines cross between staves, a stroke that does not cross is a stem. A double barline or repeat counts once. The line that opens a system ends no bar, and nor does a repeat just after the clef and key.

`layout_truth.py` gives the right layout. For a synthetic page it comes from MuseScore's SVG, which marks every staff line and barline. For a scan it is each system's staves and bars, counted on the page by eye and written in the case. `staffreader layout` scores every page and draws the found layout over it.

## Readers

A reader is a module in `staffreader/baselines` with a `NAME` and `read(image, page, work)`, which returns a MusicXML file.

- **staffreader** is this project's reader.
- **audiveris** runs Audiveris 5.11: the page rendered so its staff lines lie about 20 pixels apart, read in batch with Italian text. It runs the Flathub build unless the `audiveris` setting names another command.
- **oemer** (MIT) runs from its own environment, `venvs/oemer` in the cache, with Python 3.10, `onnxruntime==1.17.3` (CPU) and `numpy==1.23.5`. Newer onnxruntime refuses its model.

## Improving the reader

`ITERATING.md` describes the loop that improves the reader: fixed test sets, one scorecard, a diagnosis of weak and strong slices, paired experiments, and a gate against regressions. Its commands are `staffreader testset`, `staffreader evaluate`, `staffreader experiment` and `staffreader diagnose`. `experiments/` holds the scripts of the runs behind the results above.

## Long jobs

Rendering, pairing scans with their transcriptions, and training run for tens of minutes or hours, often two at a time. These rules keep a long job from thrashing the machine while it looks alive.

- **Watch every job with `uv run staffreader progress`.** Every job of ten minutes or more reports its progress there. The readout's first line shows the machine's free memory, swap and memory pressure, and warns when jobs are short of memory. A running job that has not advanced for ten minutes and for four times its usual gap is marked `STALLED?`.
- **Keep memory bounded.** A job works through its input in batches and keeps only what it has finished: `realign.readings` cuts and reads eight pages at a time, never a whole volume. Size worker pools to the memory each worker needs, not to the cores.
- **Cap picture sizes at the source.** A scan's pages are drawn no larger than 6,000 pixels on a side (`real.MAX_SIDE`). Some scans declare pages over a meter tall, and at 300 dpi one such page takes gigabytes once flattened.
- **Track a job by its process number.** Start a job with its number written to a file (`... & echo $! > job.pid`) and wait on that file (`tail --pid=$(cat job.pid) -f /dev/null`). Never find a job by searching command lines, which finds the search itself.
- **Stop the whole tree.** A job's worker pool outlives a killed parent. Stop the workers by their numbers (`pgrep -P <parent>`) as well as the parent.
- **Cache per stage and version.** Page layouts and model readings are cached beside each scan, per layout version, so a stopped job resumes where it stopped.

## Reading the web politely

staffreader reads IMSLP's catalog, the Internet Archive and the Bavarian State Library (MDZ) within each one's published rules, one request at a time, with a contact in its User-Agent (the `contact` setting), and stops at the first refusal.

`imslp.py` is the only way this project talks to IMSLP. IMSLP's robots.txt forbids automated access to `/images/` and `/imglnks/` (where every PDF is served), to `Special:` and `File:` pages and to `/index.php`, and it asks for two seconds between requests. So the module:

- reads only through `api.php`;
- waits five seconds between requests, across every process on the machine;
- sends a contact User-Agent and `maxlag`;
- halts at the first refusal, leaving `imslp.STOPPED` in the cache, which a person removes once they have looked;
- never downloads a score.

## Publishing

Nothing private goes into the code or its history: no secret, no personal address, no path on someone's machine. `uv run python scripts/check_clean.py` checks the tracked files, and `--history` checks every commit, against a private deny list kept outside the repository (`private-strings.txt` in the cache).

## License

The code and the model weights are licensed under the Apache License 2.0 (`LICENSE`). The answer keys and the published data are dedicated to the public domain under CC0 1.0 (`answers/LICENSE`).
