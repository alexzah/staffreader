# The models

`staffreader fetch` downloads two trained models from the `models-v1` release into the cache, where the reader looks for them, and checks each against its checksum (`models.py`).

| Model | What it reads | Parameters | File |
|---|---|---|---|
| `v3-mix` | each staff's notes and marks | 26.0M | `v3-mix.pt`, 104 MB |
| `text-v4` | lyrics and instrument names | 4.4M | `text-v4.pt`, 18 MB |

Both are PyTorch state dictionaries, loaded by `model.load` and `text.load`. [How it works](how-it-works.md) describes the networks.

## Intended use

Reading scans of printed scores into MusicXML that a person then checks bar by bar against the scan. A reading is a first draft: on the test pages, about nine bars in ten need no change in chamber music, songs and orchestral scores, and about six in ten in solo piano. The models are not meant for handwriting, tablature, or any setting where a reading is used unchecked.

## Training data

Only data free of conditions beyond attribution was used, so the models carry no conditions beyond their own license. Nothing licensed for non-commercial use only was trained on.

- **Engraved staves.** Scores from [PDMX](https://zenodo.org/records/14648209) marked CC0 or public domain, engraved by MuseScore 4 in seven music fonts, cut into staves and labeled from the MusicXML they were engraved from: about 836,000 staves of one-staff and two-staff parts, including sets made for weaknesses (bass parts in tenor clef, old vocal C clefs, players' parts with multi-bar rests). In training, each engraved staff is made to look printed and scanned: uneven ink, blur, noise, specks and paper that is not white.
- **Real printed staves.** Public-domain scans paired with transcriptions made from them: OpenScore's CC0 string quartets (46,830 staves), Lieder (21,006) and orchestral scores (22,059), and PDMX transcriptions that name the IMSLP scan they were made from (12,169). Each scan was paired with its transcription by content, system by system: the reader read the scan, and each system was placed at the song and bar whose notes it shared most. Pairs that disagreed were left out.
- **Text lines.** About 800,000 lines of lyrics and instrument names cut from the engravings, about 70,000 lines of directions, and about 6,500 real lyric lines and real staff labels cut from the paired scans.

Works on the test ladder, and the volumes held out for testing, never train; they are kept out of the engraved data by title and by melody.

## Training

The staff reader was trained on one RTX 3080 (10 GB) in five overnight runs, each continuing from the model before, about 30 hours in all, with mixed precision. Each run's recipe was chosen by short paired experiments against a control ([improving.md](improving.md)). `v3-mix`, the last, trained for 6 hours on the engraved and real sets above. A newer run, `v4-mix`, measured no better overall, so `v3-mix` stays the default.

## How well they read

See [results.md](results.md).

## License

Apache License 2.0, as the code.
