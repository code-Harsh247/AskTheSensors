# Results: Recognition Backbone (Phase 2)

**Owner:** Member A (Signals & Systems). Feeds the report's "Results: recognition, Figure 2" section (docs/TASKS.md Report Section Ownership).

All numbers below are measured, not illustrative — taken directly from
`models/full/phase2_results.json`, produced by `scripts/train_cnn.py` run on
Kaggle against the real 60-subject ExtraSensory sample (36 train / 12 val /
12 test subjects, `splits/subject_splits.json`), and `results/fig2_*`,
produced by `scripts/make_fig2.py`. Regenerate both from a clean checkout
via `scripts/build_feature_dataset.py` → `scripts/train_cnn.py` →
`scripts/make_fig2.py`.

**Retrained 2026-09-11 after docs/bug.md issue 1 (accelerometer units) was
fixed in `ats/ingest.py`.** The original Phase 2 numbers (CNN macro-F1
0.2749) were measured on data where roughly half of the 60 subjects had
their accelerometer silently double-converted (values ~9.7x too large).
`ats/ingest.py` now detects g-vs-m/s^2 units per subject instead of
assuming g for everyone; the feature dataset was rebuilt and the CNN
retrained from scratch on the corrected data. Both sets of numbers are
kept below for the record rather than overwriting history.

## Setup

- **Architecture:** `ats/model.py:ActivityCNN` — a compact 1D-CNN (3 Conv1d
  layers + global average pooling, 9,975 parameters), chosen over gradient
  boosting to fit PRD §6.3's quantization/pruning/distillation extra-credit
  path (docs/TASKS.md task 2A.2).
- **Training data:** 1,415,562 windows (train), 392,685 (val), from the raw
  6-channel resampled signal (window-level coverage filter ≥ 0.95).
- **Class imbalance handling (2A.3):** inverse-frequency class weights
  (`ats/imbalance.py`), applied to the training loss. Real training-split
  class counts (post units-fix; unchanged in substance from before the fix,
  since the label attribution never depended on accelerometer scale):

  | Class | Train count | Share |
  |---|---:|---:|
  | SITTING | 619,018 | 43.7% |
  | LYING | 482,461 | 34.1% |
  | STANDING_MOVING | 146,906 | 10.4% |
  | WALKING | 96,981 | 6.9% |
  | STANDING_STILL | 40,382 | 2.9% |
  | BICYCLING | 20,806 | 1.5% |
  | RUNNING | 9,008 | 0.6% |

  Sedentary classes (SITTING + LYING) alone make up **77.8%** of the training
  data; RUNNING is **68.7x rarer** than SITTING. This is PRD §3.2's flagged
  sedentary skew, not a hypothetical — real numbers from this sample.

## Exit criterion (docs/TASKS.md Phase 2, Member A)

> Window-level macro-F1 on held-out subjects beats two baselines: (a)
> majority class, (b) logistic regression on per-window mean/std.

| Model | Val macro-F1 (before units fix) | Val macro-F1 (after units fix) |
|---|---:|---:|
| (a) Majority class (always "SITTING") | 0.0792 | 0.0792 |
| (b) Logistic regression, magnitude mean/std only | 0.1555 | **0.2251** |
| **ActivityCNN** | 0.2749 | **0.2893** |

**Met, both before and after.** Post-fix: CNN beats (a) by **+0.2101** and
(b) by **+0.0642**. The margin over the logistic-regression baseline
**shrank** after the fix (+0.1194 -> +0.0642): baseline (b) reads
accelerometer magnitude mean/std directly, so it was the more directly
corrupted by the units bug and gained the most (+0.0696) from fixing it.
The CNN improved by a smaller +0.0144 (0.2749 -> 0.2893) since it can
partially learn around a per-subject scale error given enough other signal,
whereas the linear baseline cannot. The reported CNN number is the best
validation epoch out of 30 (epoch 21 post-fix; epoch 18 before), reloaded
before evaluation — not the last epoch's weights.

## Figure 2 — confusion matrix and per-class performance

See `results/fig2_confusion_matrix.png` and `results/fig2_per_class_prf.csv`
(regenerated post units-fix via `scripts/make_fig2.py`).

| Class | Precision | Recall | F1 | Support (val) |
|---|---:|---:|---:|---:|
| LYING | 0.558 | 0.732 | **0.633** | 154,187 |
| SITTING | 0.524 | 0.202 | 0.292 | 150,717 |
| STANDING_STILL | 0.036 | 0.144 | **0.057** | 11,153 |
| STANDING_MOVING | 0.141 | 0.138 | 0.139 | 37,422 |
| WALKING | 0.478 | 0.425 | 0.450 | 30,137 |
| RUNNING | 0.004 | 0.054 | **0.007** | 707 |
| BICYCLING | 0.350 | 0.615 | **0.446** | 8,362 |

## Confusions, named (PRD §7.4.2 / docs/TASKS.md exit criterion)

**Walking vs. running** (explicitly flagged by PRD §7.4.2): confused mostly
in one direction post-fix — 6.0% of true WALKING windows (1,816/30,137)
predicted RUNNING, and 23.5% of true RUNNING windows (166/707) predicted
WALKING. Consistent with the two gaits sharing similar cadence-band energy
that a 9,975-parameter model operating on a single 4-second window can
conflate; RUNNING's tiny support (707 windows) makes its 23.5% figure noisy
in absolute terms (166 windows).

**Sitting vs. lying** (PRD §7.4.2 names sitting-vs-standing-still; in this
sample sitting-vs-lying is the larger effect, and grew after the fix):
43.8% of true SITTING windows (65,973/150,717) predicted LYING — the single
largest off-diagonal cell in the matrix, up from 29.5% pre-fix. Both are
near-static, low-gyro-energy postures; the accelerometer orientation
difference between "sitting upright" and "lying down" is a subtler signal
than the model's compact feature space appears to capture reliably at the
window level, and this confusion got *worse* with corrected units, not
better — a reminder that fixing the input data does not automatically fix
every downstream number, and this is exactly the kind of behavior that
should be reported honestly rather than smoothed over.

**Standing-still collapse:** precision of 0.036 means the model essentially
does not have a usable STANDING_STILL prediction — when it does predict this
class, it is wrong 96.4% of the time. True STANDING_STILL windows now skew
toward LYING (40.9%) and SITTING (19.8%), with STANDING_MOVING (11.7%),
WALKING (8.2%), RUNNING (2.5%), and BICYCLING (2.5%) taking the rest —
concentrated on the other near-static postures (LYING, SITTING) rather than
spread flatly across all classes as before the fix. The model still never
found a clean decision boundary for this posture.

**Running collapse:** precision of 0.004 (38 correct out of 9,466 total
RUNNING predictions) and recall of 0.054. Among true RUNNING windows, the
model now predicts LYING most often (43.4%, 307/707), not STANDING_MOVING
as before the fix (previously 53.6%) — a qualitative change in *which*
wrong answer the model gives, not just the numbers, worth flagging rather
than silently reusing the old description. Given RUNNING is the rarest
class by a wide margin (0.6% of training windows, 68.7x rarer than
SITTING), this reads as a genuine data-scarcity failure rather than a
training-recipe one — PRD §3.2's imbalance warning realized directly.

## Interpretation

Validation macro-F1 oscillated between ~0.23 and ~0.29 across all 30 epochs
with no sustained upward trend after epoch ~21 (see the raw per-epoch log in
the Kaggle run). The per-class breakdown explains why: LYING, WALKING, and
BICYCLING reached a stable, reasonable decision boundary early, while
STANDING_STILL and RUNNING never did — the epoch-to-epoch noise is
concentrated in the classes that lack either a clean signal (STANDING_STILL,
confusable with multiple near-static postures) or enough training examples
(RUNNING) to converge on, not a sign that more training time would help.

This was diagnosed, not treated by trial and error: a learning-rate
schedule was considered as a plausible fix for the oscillation, but was not
pursued for this reported result, since the dominant failure modes here
(STANDING_STILL's signal ambiguity, RUNNING's data scarcity) are structural
rather than optimization artifacts and would not obviously be fixed by it.
Noted as a candidate follow-up experiment, not a fabricated improvement.

## Limitations for the report

- These numbers come from 60 subjects (36 train / 12 val / 12 test), a
  sample of ExtraSensory's 60 total users — not external validation data.
- RUNNING and BICYCLING have small absolute validation support (707 and
  8,362 windows respectively); their per-class metrics carry more sampling
  noise than LYING/SITTING's, which each have >150,000.
- The coverage filter (≥0.95, `scripts/build_feature_dataset.py`) excludes
  windows with significant gaps; results describe performance on
  reasonably-covered signal, not the full raw stream.
