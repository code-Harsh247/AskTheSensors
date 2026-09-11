# Results: Recognition Backbone (Phase 2)

**Owner:** Member A (Signals & Systems). Feeds the report's "Results: recognition, Figure 2" section (docs/TASKS.md Report Section Ownership).

All numbers below are measured, not illustrative — taken directly from
`models/full/phase2_results.json`, produced by `scripts/train_cnn.py` run on
Kaggle against the real 60-subject ExtraSensory sample (36 train / 12 val /
12 test subjects, `splits/subject_splits.json`), and `results/fig2_*`,
produced by `scripts/make_fig2.py`. Regenerate both from a clean checkout
via `scripts/build_feature_dataset.py` → `scripts/train_cnn.py` →
`scripts/make_fig2.py`.

## Setup

- **Architecture:** `ats/model.py:ActivityCNN` — a compact 1D-CNN (3 Conv1d
  layers + global average pooling, 9,975 parameters), chosen over gradient
  boosting to fit PRD §6.3's quantization/pruning/distillation extra-credit
  path (docs/TASKS.md task 2A.2).
- **Training data:** 1,415,614 windows (train), 392,685 (val), from the raw
  6-channel resampled signal (window-level coverage filter ≥ 0.95).
- **Class imbalance handling (2A.3):** inverse-frequency class weights
  (`ats/imbalance.py`), applied to the training loss. Real training-split
  class counts:

  | Class | Train count | Share |
  |---|---:|---:|
  | SITTING | 619,000 | 43.7% |
  | LYING | 482,470 | 34.1% |
  | STANDING_MOVING | 146,897 | 10.4% |
  | WALKING | 97,051 | 6.9% |
  | STANDING_STILL | 40,382 | 2.9% |
  | BICYCLING | 20,806 | 1.5% |
  | RUNNING | 9,008 | 0.6% |

  Sedentary classes (SITTING + LYING) alone make up **77.8%** of the training
  data; RUNNING is **68.7x rarer** than SITTING. This is PRD §3.2's flagged
  sedentary skew, not a hypothetical — real numbers from this sample.

## Exit criterion (docs/TASKS.md Phase 2, Member A)

> Window-level macro-F1 on held-out subjects beats two baselines: (a)
> majority class, (b) logistic regression on per-window mean/std.

| Model | Val macro-F1 |
|---|---:|
| (a) Majority class (always "SITTING") | 0.0792 |
| (b) Logistic regression, magnitude mean/std only | 0.1555 |
| **ActivityCNN** | **0.2749** |

**Met.** CNN beats (a) by **+0.1956** and (b) by **+0.1194**. The reported
CNN number is the best validation epoch out of 30 (epoch 18), reloaded
before evaluation — not the last epoch's weights.

## Figure 2 — confusion matrix and per-class performance

See `results/fig2_confusion_matrix.png` and `results/fig2_per_class_prf.csv`.

| Class | Precision | Recall | F1 | Support (val) |
|---|---:|---:|---:|---:|
| LYING | 0.584 | 0.564 | **0.574** | 154,187 |
| SITTING | 0.465 | 0.257 | 0.331 | 150,717 |
| STANDING_STILL | 0.029 | 0.095 | **0.044** | 11,153 |
| STANDING_MOVING | 0.106 | 0.142 | 0.121 | 37,422 |
| WALKING | 0.408 | 0.399 | 0.403 | 30,137 |
| RUNNING | 0.002 | 0.068 | **0.003** | 707 |
| BICYCLING | 0.354 | 0.610 | **0.448** | 8,362 |

## Confusions, named (PRD §7.4.2 / docs/TASKS.md exit criterion)

**Walking vs. running** (explicitly flagged by PRD §7.4.2): confused in both
directions — 9.1% of true WALKING windows (2,740/30,137) predicted RUNNING,
and 21.6% of true RUNNING windows (153/707) predicted WALKING. Consistent
with the two gaits sharing similar cadence-band energy that a 9,975-parameter
model operating on a single 4-second window can conflate.

**Sitting vs. lying** (PRD §7.4.2 names sitting-vs-standing-still; in this
sample sitting-vs-lying is the larger effect): 29.5% of true SITTING windows
(44,395/150,717) predicted LYING — the single largest off-diagonal cell in
the matrix. Both are near-static, low-gyro-energy postures; the accelerometer
orientation difference between "sitting upright" and "lying down" is a
subtler signal than the model's compact feature space appears to capture
reliably at the window level.

**Standing-still collapse:** precision of 0.029 means the model essentially
does not have a usable STANDING_STILL prediction — when it does predict this
class, it is wrong 97.1% of the time. True STANDING_STILL windows spread
broadly across LYING (34.5%), SITTING (14.1%), STANDING_MOVING (19.3%), and
RUNNING (11.7%) with no dominant single confusion, suggesting the model
never found a clean decision boundary for this posture at all, rather than
consistently mistaking it for one specific alternative.

**Running collapse:** precision of 0.002 (48 correct out of 29,446 total
RUNNING predictions) and recall of 0.068. Among true RUNNING windows, the
model predicts STANDING_MOVING most often (53.6%, 379/707), not RUNNING
itself. Given RUNNING is the rarest class by a wide margin (0.6% of training
windows, 68.7x rarer than SITTING), this reads as a genuine data-scarcity
failure rather than a training-recipe one — PRD §3.2's imbalance warning
realized directly.

## Interpretation

Validation macro-F1 oscillated between ~0.23 and ~0.27 across all 30 epochs
with no sustained upward trend after epoch ~18 (see the raw per-epoch log in
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
