# Using staffreader

## Installing

You need [uv](https://docs.astral.sh/uv/), which fetches Python 3.13, and poppler's `pdftoppm` for reading PDFs. An NVIDIA GPU makes reading fast; the CPU works, more slowly. Engraving test pages and training data also needs MuseScore 4.

```bash
git clone https://github.com/alexzah/staffreader && cd staffreader
uv sync
uv run staffreader fetch          # the trained models and the answer keys
```

`uv sync --group dev` adds pytest; `uv run pytest` runs the tests, which skip themselves where MuseScore, the answer keys or the scans are missing.

## Reading

```bash
uv run staffreader read edition.pdf --page 12 -o page12.musicxml     # one page of a PDF
uv run staffreader read page.png                                    # a picture of a page
uv run staffreader read edition.pdf --pages 12-15 -o work.musicxml   # a run of pages, as one work
```

A run of pages is read as one work: the parts are named on the first page and matched on the rest by their labels, and a part a page leaves out rests there. `--names` names each system's staves by hand, for a page whose labels the reader gets wrong; `--layout` and `--picture` also write the staves, systems and bars it found and the straightened page they were found on.

A reading is a first draft. Open it in MuseScore beside the scan and check it bar by bar.

## Settings

staffreader keeps its work (models, renders, data sets and runs) in `~/.cache/staffreader`. Settings go in `~/.config/staffreader/settings.toml`, and any setting can be given instead as an environment variable, `STAFFREADER_<NAME>`:

```toml
cache = "~/.cache/staffreader"   # models, renders, data sets and runs
scans = "~/scores/imslp"         # printed scans, named as IMSLP names them
pdmx = "~/datasets/pdmx"         # the PDMX dataset, for engraving training data
contact = "you@example.org"      # sent with every request to IMSLP, the Internet Archive and the MDZ
mscore = "mscore"                # MuseScore 4
device = "cuda"                  # or "cpu"
```

`settings.py` lists every setting.

## Commands

| Command | What it does |
|---|---|
| `read` | read a page, or a run of pages, into MusicXML |
| `fetch` | download the models and the answer keys (`--models`, `--answers` for one) |
| `suite` | read the test ladder with each reader and write the scoreboard ([evaluation.md](evaluation.md)) |
| `layout` | find and score the staves, systems and barlines of the ladder's pages |
| `compare` | score one reading against one answer |
| `imslp` | look up an IMSLP file: its edition, pages, copyright, and the link to download it by hand |
| `scans` | fetch the scans OpenScore's quartets were transcribed from, from the Internet Archive |
| `progress` | how far every long job has got (`--watch` to follow it) |
| `testset`, `evaluate`, `diagnose`, `experiment`, `stale` | the loop that improves the reader ([improving.md](improving.md)) |

`uv run staffreader <command> --help` gives each command's options.
