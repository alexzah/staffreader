# Improving the reader

Once the reader is functionally complete, it improves by one repeated loop. Each turn finds where the reader is weak, changes its training data or pipeline to fix that, and proves that the change helps without breaking something else.

1. **Measure** the current model on fixed test sets with one scorecard.
2. **Diagnose** where it is weak and where it is strong, by slicing every reading by what was on the page.
3. **Propose** a fix for the weakness that costs the most correction time, reasoned from the evidence.
4. **Test** the fix in a short paired experiment against an unchanged control.
5. **Gate** it: the target improves beyond noise, and nothing else gets worse beyond noise.
6. **Batch** the fixes that pass into a long run, which a maintainer approves, and measure again.

The rest of this document says how each step works and which command does it. `README.md` says how the tool itself works.

```bash
uv run staffreader testset                                   # build the reserved test sets (once, and when the reserved scans change)
uv run staffreader evaluate <model.pt> --against <folder>    # steps 1-3: measure, diagnose, compare; writes runs/eval-<name>/index.html
uv run staffreader experiment <spec.yaml>                    # step 4: train a control and variants, evaluate and compare each
uv run staffreader diagnose <folder> --against <folder>      # draw a diagnosis page again from its records
```

## 1. Test sets

Each test set has one job. The reader never trains on any of them.

- **The ladder** (`cases/*.yaml`, read with `staffreader suite`). Its pages are checked by eye: real scans of the edition a transcription was made from, and engravings of the same bars. It is small and unbiased, and it is the final word. Every weakness that matters gets a page here. A case's whole volume leaves training automatically (`train.samples`).
- **Reserved real volumes** (`data/test-<music>`, built with `staffreader testset`). Every page of the scans that never train: each ladder case's scan and the volumes `reserved.yaml` adds. Today they are four quartet scans, the Arabesque, Mozart 40, Brahms 3 and Beethoven 1, and two Lieder volumes, giving test sets for solo piano, voice and piano, quartet and orchestra. Each system is placed by its content, or by the music's order between placed systems, and kept however well a model reads it, so the sets give broad real coverage without flattering any model.
- **Held-out training data** (one real work in ten, one rendered score in fifty, `train._held_out`). It is large and fast, and it suits trends and sanity checks. It is not for judgment: the Lieder and orchestral pairs were kept only where an earlier model read them well, so they flatter.
- **Text lines** for the text reader: held-out rendered lines (`text.lines`), and real lyric lines (`data/test-lyrics`) cut under the reserved voice staves as the reader cuts them (`testsets.lyric_lines`). A staff gives lines only where the rows found under it are as many as its verses, so this set measures reading a line, not finding it. Finding rows is measured by the lyrics layer of the ladder and the reserved voice set.

## 2. The scorecard

`staffreader evaluate` writes the scorecard for one model over every test set. Each figure carries a 90% interval from a bootstrap over pages or staves, so a difference within the noise is shown as one.

- **Notes:** note F1 at the core level (pitch and onset), and the share of bars read exactly.
- **Correction cost:** the edits a person would make to correct a page (`diagnose.edits`), the yardstick that matters to whoever uses the reader. A wrong clef or key signature counts as one edit, not one per note it moves. A wrong note, a missed note, an extra note or a wrong mark counts one each.
- **Mistakes by kind**, as `metrics.classify` names them.
- **Marks:** F1 for each family of the inventory (`metrics.family`).
- **Text:** lines read exactly and characters wrong, for lyrics and for instrument names apart.
- **Layout:** pages laid out exactly (`staffreader layout`).
- **Speed:** seconds a page.

Every evaluation appends its scorecard to `runs/scorecards.jsonl`, one line each, so the trend across turns of the loop reads from one file.

## 3. Diagnosis

`staffreader evaluate <model>` keeps a record of every staff it reads, and its diagnosis (`diagnose.py`) slices those records by what each staff holds and where it came from:

- the kind of music, the source (real or rendered), the edition and the font;
- the clef, key, time, density (notes a bar), voices, chords, tuplets, grace notes, ledger lines, and the families of marks;
- the staff's role: a grand staff's upper or lower staff, a voice, an orchestral family.

For each slice it gives the scorecard, then ranks slices by **impact**: how far a slice falls below the reader's average, times how often the slice occurs in the music we want to read. A rare clef read badly matters less than a common figure read slightly badly. It also writes a **gallery** of the worst staves of each weak slice, with the picture, the right answer and the reading side by side. Whoever chooses a fix looks there first, since a number cannot say why a staff went wrong.

`staffreader evaluate <model> --against <folder>` compares two models on the same staves and pages. It shows each slice's change with its interval, and marks regressions.

## 4. From a weakness to a fix

The fix comes from studying the evidence, not from a list of known fixes. A list would steer every weakness toward a fix that worked before, and the right fix is often one the project has not needed yet: a change to the model, the data, the layout, the decoding, the assembly or the harness itself.

Before any fix, rule out the test itself. A weak slice is often a wrong answer key, not a weak reader: on 2026-10-15 half the wrong orchestral bars were two wind players on one staff labeled with one player, the orchestral labels carried an analyst's theme letters as lyrics, and a fifth of the real lyric lines sat under another volume's labels through a file-name collision. Each was found by looking at the worst staves' pictures beside their labels, and each would have sent a fix after a weakness that was not there.

Whoever proposes a fix, usually the agent, writes down:

- **What is wrong,** from the gallery and from wrong bars looked at by eye, not from the numbers alone. A number says where the reader is weak, never why.
- **Why it goes wrong:** the cause, with the evidence for it, and what else could explain the same evidence. Where two causes fit, the experiment is designed to tell them apart.
- **The fix,** and why it addresses that cause and not only its symptom.
- **What it should change:** the slice expected to improve and by about how much, and the slices it could hurt.
- **How it is tested:** the experiment that shows it (step 5), or a ladder page that shows it for a fix outside training.

A fix must be general. A change made only to pass one ladder page is overfitting: it has to help the reserved volumes too. The plan's record of past turns (the E-series experiments) is history to learn from, not a menu.

## 5. Testing a fix

Every fix is tested in a short paired experiment before it reaches a long run. `staffreader experiment <spec.yaml>` runs one from a written spec:

- a **control** and one or more **variants**, all fine-tuned from the same model for the same number of steps, so a busy machine cannot favor one;
- the same test sets for each, read with the same decoder;
- one variant changes one thing, so its effect has one cause (as E8c separated the marks from the data).

It writes each run's evaluation, every variant's compared with the control's, and a summary in `runs/experiment-<name>.json`. The spec's form is in `evaluate.experiment`:

```yaml
name: e9-tuplets
steps: 3500
runs:
  control: {data: [r2-single-quarter, r2-grand, real3-v2*8], tasks: true}
  variant: {data: [r2-single-quarter, r2-grand, real3-v2*8], tasks: true,
            boost: {tuplets: 3}, target: {tuplets: tuplets}}
```

A spec can also show chosen staves more often in training, when the proposal calls for that. A run's `boost` names slices by the diagnosis's own features and how many times to show each staff in them (`train.boost_count`): `{tuplets: 3}` shows every staff with a tuplet three times, and `{clef: {C4: 4}}` every tenor-clef staff four. A staff in several boosted slices takes the largest factor. Its `target` names the slice the variant aims to improve, which the gate checks.

## 6. The gate

A variant passes when:

- the slice it targets improves by more than its interval;
- no slice of the reserved volumes or the ladder gets worse by more than its interval;
- overall note F1 and bars exact on the reserved volumes do not fall.

`staffreader experiment` applies the first two rules itself (`evaluate._gate`) and writes each variant's verdict: passes, fails because something got worse elsewhere, or fails because its target did not improve. A variant that helps its target and hurts elsewhere goes back to step 3 with that finding, as marks did (E8b, then E8d).

## 7. The long run

Fixes that pass are batched into one long run, which a maintainer approves, since it holds the GPU for hours. After it, `staffreader diagnose --against` the model before gives the turn's result, recorded in the project's history of runs. The next turn starts from the new diagnosis.

### The control stays current

The model before (the control) is measured once, before or beside the run, never after it, so the run's results never wait on it: on the CPU while the run trains (`STAFFREADER_DEVICE=cpu`), or before the run starts. Its measurement must still describe what a measurement made at the run's end would: the same decoder on the ladder, the same test labels, the same scoring code. Work done while a run trains often changes one of them (a decoding fix, a corrected label, a new mask), and then the control has gone stale.

So whenever reading, scoring or test labels change during a run, bring the control up to date at once, while there is time:

- test labels or scoring changed: score its stored readings again, `uv run python experiments/rescore_eval.py <name>` (minutes, on the CPU);
- the decoder changed: read its ladder again, `STAFFREADER_MODEL=<model> staffreader suite --reader staffreader --out runs/eval-<name>/ladder` (on the CPU while the GPU trains);
- staff reading changed (model.py, staffcrop.py, layout.py): measure it again, `staffreader evaluate`.

`staffreader stale runs/eval-<name>` says whether a control is current and what to do if not; run it after every such change and before calling a control ready. `staffreader evaluate --against` runs the same check and writes `stale-control.txt` beside the results when it fails, but by then it is late: the comparison is already unfair.

## Rules

- Compare runs at a fixed number of steps, never a fixed time.
- Never train on a test page. A ladder case's volume and every reserved volume are kept out of training by the code, not by care.
- Look at the gallery before choosing a fix, and look at wrong bars by eye before trusting a metric. When the reading looks right and the label does not, fix the test set first and measure again.
- Pilot every new kind of data or setting for minutes before a long run (CONVENTIONS.md, "A long job shows its progress and keeps its memory bounded").
