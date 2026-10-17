# Results

Measured on 2026-10-17 with the published models, `v3-mix` and `text-v4`. [How it is measured](evaluation.md) describes the test pages and the scoring. Notes are scored as F1 at pitch and onset; "bars" is the share of bars needing no correction.

## The ladder: printed pages

Each page is from the edition its answer key was transcribed from. Audiveris 5.11 reads the same pictures; a page it could not read at all (it wrote no MusicXML, or ran out of memory) is left out of its averages and counted beside them.

| Music | Pages | staffreader | Audiveris 5.11 |
|---|---|---|---|
| String quartets | 4 | 0.99 notes, 95% of bars | 0.58 notes, 34% of bars; no reading of 2 pages |
| Voice and piano | 4 | 1.00 notes, 97% of bars | 0.79 notes, 56% of bars |
| Orchestra | 8 | 0.98 notes, 92% of bars | 0.83 notes, 66% of bars; no reading of 1 page |
| Solo piano | 5 | 0.91 notes, 64% of bars | 0.33 notes, 2% of bars |

Page by page, bars needing no correction:

| Page | staffreader | Audiveris |
|---|---|---|
| Haydn, "Lark" Quartet, I, p. 1 | 100% | 25% |
| Mozart, "Dissonance" Quartet, I, p. 1 | 95% | no reading |
| Haydn, "Emperor" Quartet, II, p. 12 | 96% | no reading |
| Boccherini, Quintet in E major, Minuetto (Ricordi), p. 22 | 90% | 43% |
| Beethoven, *Mollys Abschied* | 95% | 45% |
| Beethoven, *Maigesang* | 100% | 70% |
| Brahms, *Auf dem See* | 94% | 48% |
| Brahms, *Regenlied* | 97% | 59% |
| Mozart, Symphony No. 40, I, pp. 1–6 | 89–99% | 67–83% |
| Brahms, Symphony No. 3, p. 11 | 86% | no reading |
| Beethoven, Symphony No. 1, p. 10 | 79% | 27% |
| Debussy, Arabesque No. 1, pp. 1–5 | 38–100% | 0–8% |

Every case also has the same bars engraved by MuseScore in a font the reader never trained on, clean and made to look scanned. Over those 42 pictures staffreader reads 0.96 of notes and 92% of bars, and Audiveris 0.81 and 62%.

## The reserved test sets: whole scanned volumes

Every page of scanned volumes that never trained, placed against their transcriptions. These are larger than the ladder and checked less closely, so they measure breadth.

| Music | Staves | Notes | Bars | Edits a staff |
|---|---|---|---|---|
| String quartets | 390 | 0.99 | 96% | 1.6 |
| Voice and piano | 475 | 0.96 | 90% | 4.3 |
| Orchestra | 3,000 | 0.94 | 86% | 5.8 |
| Solo piano | 150 | 0.95 | 84% | 19.2 |

"Edits a staff" counts the changes a person would make to correct a staff: a wrong clef or key signature counts once, and each wrong, missed or extra note or mark once.

## Marks

F1 for the commonest families of marks on the reserved test sets. Dynamics and hairpins in voice and piano are scored as the part is assembled, since they print between the piano's staves.

| Marks | Quartets | Voice and piano | Orchestra | Solo piano |
|---|---|---|---|---|
| Slurs | 0.97 | 0.96 | 0.91 | 0.69 |
| Articulations | 0.98 | 0.97 | 0.91 | 0.63 |
| Dynamics | 0.98 | 0.79 | 0.90 | 0.67 |
| Hairpins | | 0.62 | 0.54 | 0.54 |
| Words | 0.66 | 0.11 | 0.38 | 0.12 |

Fingering on printed pages, tempo marks, octave lines and endings are not yet read reliably.

## Text

The text reader reads 93% of engraved lyric and label lines exactly. On real lyric lines cut from the reserved Lieder, it reads 32% of lines exactly, with 3.0% of characters wrong: the rest mostly miss by a few characters.

## How it got here

The reader was built from 2026-10-12 to 2026-10-17. Each long run continued from the model before, and each change was tested against a control first ([improving.md](improving.md)).

| Model | Date | What changed |
|---|---|---|
| `night-mix` | 10-13 | First full training run on engraved staves and real quartet pages: 77% of held-out real bars exact, beating Audiveris on every real page it could read |
| `m4b-mix` | 10-14 | Voices and grand staves, real Lieder pages paired by content |
| `v2-mix` | 10-15 | Marks of every family and lyrics; each staff read twice, for notes and for marks |
| `v3-mix` | 10-16 | Token format 3, real piano pages paired with PDMX transcriptions, old clefs: solo piano from 80.5% to 83.6% of bars on the reserved set |
| `v4-mix` | 10-17 | Measured no better overall; `v3-mix` stays the default |
