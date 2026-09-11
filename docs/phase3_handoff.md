# Phase 3 Handoff: Real Dev Subjects for `oracle_delta.py`

**For:** Member B (Harsh). **From:** Member A. **Date:** 2026-09-11.

## Why this exists

Phase 3 ("The Swap") needs a subject with *both* a gold question set and a
real track — but the existing dev question sets (`data/questions_dev/subj_synth_{a,b,c}.json`)
are for synthetic fixture subjects with no real sensor data behind them, and
the trained model only runs on real ExtraSensory subjects. Neither of us
could close this gap alone: you don't have the raw dataset on your machine
(per your own note in `docs/TASKS.md`'s Phase 2 status), and I don't know
your question-authoring conventions well enough to invent good ones solo.

This hands you everything needed to write real gold questions yourself:
two real subjects, their full oracle and real-model tracks (already
generated, already committed — you don't need the raw dataset or the
trained model to use them), and a ground-truth summary computed with
**your own** `ats.aggregate.build_timeline` (not my reinterpretation of it),
so the bout boundaries and durations you write questions against are
exactly what your code will also produce when scoring them.

## What's here

Two real ExtraSensory subjects, chosen to cover all 7 canonical classes
between them (subject A alone is missing RUNNING and BICYCLING):

| Alias | Real subject UUID | Classes present |
|---|---|---|
| `subj_real_a` | `00EABED2-271D-49D8-B599-1D4A09240601` | LYING, SITTING, STANDING_STILL, STANDING_MOVING, WALKING |
| `subj_real_b` | `74B86067-5D4B-43CF-82CF-341B76BEA0F4` | all 7 (including RUNNING, BICYCLING) |

Files, matching the naming `ats/eval/dev.py` and `scripts/oracle_delta.py`
already expect (subject = question-set filename stem):

```
tests/fixtures/track_subj_real_a.jsonl                    -- oracle (ground truth) track
tests/fixtures/track_subj_real_b.jsonl                    -- oracle (ground truth) track
tests/fixtures/real_model_tracks/track_subj_real_a.jsonl  -- real ActivityCNN track
tests/fixtures/real_model_tracks/track_subj_real_b.jsonl  -- real ActivityCNN track
```

The oracle tracks are already in `tests/fixtures/` (the default
`--oracle-dir`), following the same `track_<subject>.jsonl` convention as
the synthetic ones. The real tracks are in a new `real_model_tracks/`
subfolder — pass it as `--real-dir` (there's no sensible default for it
anyway, since it didn't exist before).

**All you need to do is write** `data/questions_dev_v2/subj_real_a.json` and
`data/questions_dev_v2/subj_real_b.json`, following the exact format your
existing `subj_synth_*.json` files use.

**Why `data/questions_dev_v2/` and not `data/questions_dev/`:** your own
note in `docs/TASKS.md`'s Phase 1 "Artifacts crossing the boundary" already
calls for exactly this — `data/questions_dev/` is frozen, later additions go
to `data/questions_dev_v2/`. It's also a practical necessity, not just
following convention: `scripts/oracle_delta.py`'s `--real-dir` requires
*every* subject with a question set to have a matching real-model track,
and the synthetic subjects don't have (or need) one. Pointing
`--questions-dir` at a directory containing only the two real question sets
keeps the real-track run from tripping over the synthetic ones. (Verified —
running it against the mixed `data/questions_dev/` directory throws
`FileNotFoundError: no track for subject subj_synth_a`, exactly this
reason.) I've added `!data/questions_dev_v2/` to `.gitignore` already.

## Ground truth summary

Computed via `ats.aggregate.build_timeline(oracle_track)` — your own
minute-attributed-bout logic, not a hand reinterpretation. Full detail
(every bout, every gap, every transition) is in the tracks themselves;
this is what a human needs to write correct questions quickly.

### subj_real_a (00EABED2) — span 0 to 639,994s

| Activity | Total duration | Bouts |
|---|---:|---:|
| SITTING | 66,022s | 129 |
| LYING | 39,358s | 48 |
| STANDING_MOVING | 11,425s | 13 |
| WALKING | 9,660s | 17 |
| STANDING_STILL | 109s | 1 |

184 gaps, 513,420s of total gap time (this subject's real data is sparse
relative to the full span — expected, matches the dataset's own duty cycle).

Longest bout per activity (useful for identification/verification questions
with an unambiguous answer):
- LYING: 311,941 → 322,020s (10,079s)
- SITTING: 98,240 → 107,059s (8,819s)
- WALKING: 84,680 → 87,800s (3,120s)
- STANDING_MOVING: 12,154 → 14,434s (2,280s)
- STANDING_STILL: 328,006 → 328,115s (109s) — the *only* STANDING_STILL bout
  for this subject; good for a duration/count question with a small, exact
  answer.

First few transitions (useful for onset/"when did X begin" questions):
- 9,356s: SITTING → WALKING
- 12,154s: WALKING → STANDING_MOVING
- 14,434s: STANDING_MOVING → SITTING
- 32,494s: SITTING → LYING

Sample gap (useful for the "answer interval lands inside a data gap" edge
case, same as your synthetic fixtures already test): 240 → 270s (30s gap).

### subj_real_b (74B86067) — span 0 to 504,693s

| Activity | Total duration | Bouts |
|---|---:|---:|
| SITTING | 151,119s | 1,188 |
| LYING | 144,050s | 1,074 |
| STANDING_MOVING | 51,151s | 467 |
| BICYCLING | 39,416s | 347 |
| RUNNING | 5,020s | 43 |
| STANDING_STILL | 1,848s | 21 |
| WALKING | 869s | 11 |

Far more fragmented than subject A (3,151 bouts / 3,110 gaps vs. 208/184)
— more real-world gap structure to test against, but individual bouts tend
to be shorter.

Longest bout per activity:
- LYING: 145,646 → 146,270s (624s)
- BICYCLING: 187,228 → 187,768s (540s)
- SITTING: 432,461 → 433,038s (577s)
- STANDING_MOVING: 29,033 → 29,557s (524s)
- RUNNING: 254,377 → 254,669s (292s) — clean, unambiguous running bout.
- STANDING_STILL: 249,544 → 249,784s (240s)
- WALKING: 171,016 → 171,177s (161s)

A genuine walking-vs-running comparison question has a clear answer here:
RUNNING (5,020s total) vs. WALKING (869s total) — running exceeds walking
for this subject, the opposite of the brief's own worked example, which
might make a good edge case precisely because it isn't the "expected"
direction.

First few transitions:
- 308s: STANDING_STILL → SITTING
- 2,457s: SITTING → STANDING_MOVING
- 2,866s: STANDING_MOVING → BICYCLING
- 4,527s: BICYCLING → WALKING

## Known caveats

- Both subjects had a handful of corrupted device-clock timestamps
  (`ats.ingest`'s `WARNING` messages during generation) — a whole channel
  (accelerometer or gyroscope) was dropped for the affected bursts rather
  than trusted. This is now baked into both tracks; nothing further to do,
  just don't be surprised if a burst's `feature_summary` looks sparse.
- The real-model track's predictions are **not** ground truth — they're
  what `oracle_delta.py` is *for*: comparing them against the oracle track
  reveals exactly where the real classifier disagrees, and your tool
  attributes each disagreement to recognition, aggregation, or routing.
  Don't be alarmed if the real track's predicted activity differs from the
  ground truth summary above at a given timestamp — that's the point.
- File sizes: oracle tracks are 7.9MB (subj_real_a) and 16MB (subj_real_b);
  real-model tracks are 11MB and 21MB. Committed as-is (full multi-day
  recordings) rather than trimmed, since trimming to a shorter window would
  have cut subject A's only STANDING_STILL bout and subject B's rarer
  RUNNING/BICYCLING bouts — the class diversity was the entire point of
  picking these two subjects.

## Once question sets exist

```
python scripts/oracle_delta.py --oracle-dir tests/fixtures --real-dir tests/fixtures/real_model_tracks --questions-dir data/questions_dev_v2
```

`--questions-dir` points at `data/questions_dev_v2/`, not the original
`data/questions_dev/` — see above for why. This produces the real
per-question-type delta table Phase 3's exit criteria ask for — the first
one built from an actual trained model rather than a simulated one.

**Verified working end-to-end** with a throwaway single-question test
(not committed): the command above ran cleanly and correctly attributed a
recognition-layer loss (gold "Sitting", oracle "Sitting", real model
"Lying down" — consistent with the confusion matrix in
`docs/results_recognition.md`). The mechanics are proven; only real
question content is missing.
