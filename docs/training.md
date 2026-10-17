# Data and training

This is how the training data was made and the models trained. It needs a machine with an NVIDIA GPU, MuseScore 4, the PDMX dataset and a few hundred gigabytes of disk; the published models ([model.md](model.md)) are what most readers want.

## Engraved data

`data.py` renders PDMX's CC0 and public-domain scores with MuseScore and cuts them into staves. Each staff is stored with its picture, its tokens and the MusicXML it came from, so a later token format relabels the staves without engraving them again. Scores are spread over seven music fonts and many page sizes, and sets are made for what the reader finds hard: bass parts written in tenor clef, the old vocal C clefs, players' parts with multi-bar rests, sparse fingering, words. Leland, MuseScore's default font, is kept for the engraved test pages, so the test measures fonts the reader has not seen.

## Real data

`real.py` and `realign.py` pair public-domain scans with the transcriptions made from them:

- OpenScore's string quartets, Lieder and orchestral scores, each of which names the IMSLP scan it was transcribed from;
- PDMX transcriptions whose metadata or credits name their IMSLP scan.

Most scans come from the Internet Archive's 2012 copy of IMSLP or Wikimedia Commons; the rest were downloaded from IMSLP by hand. A volume is read once by the current model, and each system is placed at the work and bar whose notes it shares most, staff by staff. A work spread over many pages is followed page by page. Systems whose reading disagrees with the transcription are left out. Placing by content recovered systems that counting bars could not: in a 135-page volume of songs, 394 of 555 systems against 50.

## Training

`train.py` trains the staff reader on a mix of sets, each with a weight (`real3*8` counts the set eight times):

```bash
uv run python -c "from staffreader import train; train.train('my-run', ['r2-single', 'real3*8'], hours=6, resume='…/v3-mix/last.pt')"
```

Each staff is seen through a window that moves and scales a little, as the layout stage's cut does on a real page, and is then made to look printed and scanned: uneven ink, blur, noise, specks, paper that is not white, a slight tilt, and sometimes a hard threshold. Batches are built by width, so a batch of short staves is not padded to the longest. `text.py` trains the text reader the same way on lines of text.

The recipes behind each published model are in `experiments/` (the `overnight_*.py` scripts), and the short experiments that chose them are described in [improving.md](improving.md).

## Long jobs

Rendering, pairing and training run for hours. These rules keep a long job from thrashing the machine while it looks alive:

- **Watch every job with `uv run staffreader progress --watch`.** Every job of ten minutes or more reports its progress there, with the machine's free memory and swap. A job that has not advanced for ten minutes and for four times its usual gap is marked `STALLED?`.
- **Keep memory bounded.** A job works through its input in batches and keeps only what it has finished: `realign.readings` cuts and reads eight pages at a time, never a whole volume. Size worker pools to the memory each worker needs, not to the cores.
- **Cap picture sizes at the source.** A scan's pages are drawn no larger than 6,000 pixels on a side (`real.MAX_SIDE`). Some scans declare pages over a meter tall.
- **Cache per stage and version.** Page layouts and model readings are cached beside each scan, per layout version, so a stopped job resumes where it stopped.

## Reading the web politely

staffreader reads IMSLP's catalog, the Internet Archive and the Bavarian State Library (MDZ) within each one's published rules, one request at a time, with a contact in its User-Agent (the `contact` setting), and stops at the first refusal.

`imslp.py` is the only way this project talks to IMSLP. IMSLP's robots.txt forbids automated access to `/images/` and `/imglnks/` (where every PDF is served), to `Special:` and `File:` pages and to `/index.php`, and it asks for two seconds between requests. So the module reads only through `api.php`, waits five seconds between requests across every process on the machine, sends `maxlag`, halts at the first refusal (leaving `imslp.STOPPED` in the cache for a person to remove), and never downloads a score.
