# Development Plan and Task Breakdown
## Ask the Sensors — CS60055 Hackathon Challenge 1

**Companion document:** [PRD.md](PRD.md) is the source of truth for *what* the system must do. This document covers *how* it gets built, *in what order*, *by whom*, and *how we know a phase is actually finished*. Where the two disagree, the PRD governs; where the PRD is silent on process, this document governs.

**Team:** two members, two tracks.

| Track | Owner | Scope |
|---|---|---|
| **A — Signals & Systems** | Teammate | Data acquisition, preprocessing, resampling, windowing, subject-wise splits, the 7-class recognition backbone, model compression, efficiency instrumentation, robustness experiments |
| **B — Reasoning & Evaluation** | Harsh | Timeline aggregation, evidence attribution, question routing, deterministic reasoning operators, SLM query interface, output-format serializer, evaluation harness and all QA metrics |

The two tracks are deliberately **vertical, not layered by phase**. Each member owns complete, nameable subsystems end to end, so the per-member contribution statement required by PRD §9.1 is a description of what actually happened rather than a reconstruction. Neither member should ever need to edit the other's core modules; the only shared files are the frozen contracts from Phase 0, the top-level README, and the report.

---

## 0. Frozen Decisions

These are settled **before** anyone writes logic, and are changed only by joint agreement in a dedicated commit. Everything downstream assumes them.

| Decision | Value | Why it is frozen |
|---|---|---|
| Time base | **Seconds from start of recording**, float, 3 decimal places | PRD §5.1 requires one consistent convention across every answer; mixing conventions silently corrupts every IoU score |
| Canonical class order | `LYING, SITTING, STANDING_STILL, STANDING_MOVING, WALKING, RUNNING, BICYCLING` | Index order appears in probability vectors, the confusion matrix, and serialized artifacts; a reorder mid-project silently mislabels everything |
| Sampling rate | **25 Hz exactly** | PRD §3.2 hard requirement |
| Headline IoU threshold | **τ = 0.5** | Pre-registered *before* seeing results, so the choice cannot be tuned to flatter our numbers. The full 0.1–0.9 sweep is reported anyway as Figure 3 |
| Duration tolerance | `max(2 × window_hop, 10% relative)` | Derived from A's Phase 1 hop, not invented. Pre-registered for the same reason |
| Count tolerance | **±1 bout**, added 2026-09-11 | PRD §7.3.2's own rule ("plus or minus one for a count"). Before this, counts were mistakenly scored with the duration tolerance, whose 4 s floor accepted a count of 5 anywhere from 1 to 9. The fix is strictly tighter, and no measured result changed |
| Target device | **Laptop CPU** — AMD Ryzen 7 4800H (8C/16T), 8 GB RAM, Windows 11 64-bit, single-process CPU inference (no GPU) | Corrected 2026-09-11: the original entry (5800H, 16GB) didn't match the machine actually producing the cost numbers (`ats/profile.py`'s task 4A.2, and Phase 5's compression/robustness sweep, both run on Member A's laptop). Chosen for reproducibility with no cross-compilation or extra hardware; PRD §6.2 explicitly allows "a laptop processor" and says clear, consistent reporting matters more than which hardware is chosen |
| Window / hop | **4.0s / 2.0s** (50% overlap), frozen 2026-09-10 | A window must fit inside one ~20s ExtraSensory recording burst with room to spare (aggregation never segments across a gap — see `ats/aggregate.py`), and 4s covers several gait cycles even at walking cadence. The 2s hop feeds directly into the duration tolerance below. See `ats/windowing.py` for the full rationale |
| Interval semantics | **Minute-attributed bouts**, decided 2026-09-11 | ExtraSensory records ~22 s of signal per labeled minute and labels per minute, so each burst stands for its whole minute: consecutive same-activity minutes merge into one bout, and a missing minute stays a real gap. Without this, "how long was she walking?" returns about a third of reality and "how many times?" counts bursts instead of bouts. Explanations state how much recorded signal backs each cited interval. Within a burst every window takes one pooled label (largest summed probability), added the same day after the oracle-delta dry run showed a few flipped edge windows surviving smoothing or claiming the unrecorded rest of their minute. An unrecorded stretch shorter than one minute is bridged rather than treated as a gap: real recordings drift a few seconds off the one-minute cadence, and such a sliver cannot hide a missing minute. Found 2026-09-11 on the first real subjects, where 1-13 s slivers had split continuous activity into hundreds of bouts. See `ats/aggregate.py` |

**Naming convention for the package:** `ats/` (Ask The Sensors).

---

## Phase 0 — Contract Freeze and Repo Skeleton

**Type:** Joint. Blocking — no other phase starts until this one exits.

**Objective:** fix the interfaces between the two tracks *before* either person writes real logic. This phase is what makes every later phase parallelizable. Do not compress it; an hour saved here costs a day at integration.

### Joint tasks

- **0.1** Create the GitHub repository; push the existing `docs/` immediately so the commit history starts now.
- **0.2** `.gitignore` excluding `data/`, `models/`, `*.csv` under `results/raw/` — PRD §3.3 forbids committing the raw dataset.
- **0.3** Environment: `pyproject.toml` (or `requirements.txt` + `environment.yml`), pinned versions, `pip install -e .` working on both machines.
- **0.4** Write `ats/contracts.py` plus JSON Schemas in `schemas/`. **These four schemas are the entire A↔B interface:**

  | Schema | Purpose | Key fields |
  |---|---|---|
  | `window_track.schema.json` | **The A→B artifact.** Per-window classifier output | `window_id`, `t_start`, `t_end`, `probs[7]` (canonical order), `coverage` (fraction of window backed by real, non-imputed samples), `feature_summary` (named dict: acc-magnitude mean/std, dominant cadence Hz, gyro energy per axis), `model_id` |
  | `answer.schema.json` | The PRD §5 output block plus machine-scoreable fields | §5 fields **in order**, plus `question_id`, `tier_inferred`, `cited_intervals[]`, `modality`, `channels` |
  | `question_set.schema.json` | Input questions | `question_id`, `text`, and an **optional** gold block |
  | `cost_report.schema.json` | PRD §6.1 numbers | `params`, `disk_mb`, `peak_rss_mb`, `latency_p50_ms`, `latency_p95_ms`, `cpu_pct`, `energy_estimate_j`, `target_device` |

  > **Why the gold block is optional:** at evaluation time we receive questions with no answers. If gold were mandatory, the eval-time input would take a different code path than everything we tested. Optional gold means the file we are handed on grading day parses through the exact same validated path as our dev set.

  > **Why `feature_summary` is in the contract:** B cites these values verbatim inside explanations (PRD §4.3, §4.4). The *key names* are therefore part of the interface — A cannot rename them unilaterally without breaking every grounded explanation.

- **0.5** Decide and record the **target device** for all efficiency measurements (PRD §6.2). Write it into §0 of this document.
- **0.6** Freeze the two entry points:
  ```
  python -m ats.answer --recording <path> --questions <path> --out <path>
  python -m ats.eval   --pred <path> --gold <path> --out <dir>
  ```
  `ats.eval` **must also** expose `evaluate(pred, gold) -> dict` as an importable function. A calls it directly for the Figure 4 and Figure 5 sweeps; shelling out for hundreds of sweep points is both slow and fragile.

  > **Extended 2026-09-11:** `ats.answer` also takes `--subject` (an ExtraSensory data directory holds many subjects), `--model` (a path to trained weights), `--save-track`, and the alternative `--track`. Exactly one of `--recording` or `--track` is required. The `--recording` path runs `ats.recognize` and is covered by `tests/test_answer_cli.py`, but it is blocked on real recordings until an unlabelled recording can be loaded ([bug.md](bug.md), issue 2).
- **0.7** Create `docs/CITATIONS.md` as a **running** log, appended in the same commit as the work it describes. PRD §9.3 requires every external resource — dataset, pretrained model, software library, borrowed code fragment — to be cited **precisely at the point of use**. Each member appends their own entries as they use things; both maintain it. An in-code comment at the use site pointing at the entry is the cheapest way to keep the two in sync.
- **0.8** Skim the four related readings in PRD §8.2 before locking the architecture, and note in `docs/CITATIONS.md` anything we borrow from them.
- **0.9** Stub `ats.answer` and `ats.eval` so both CLIs exist and parse arguments, with `NotImplementedError` bodies.

### Exit criteria

- [x] `pytest tests/test_contracts.py` passes, and specifically asserts:
  - a hand-written example of each of the four schemas round-trips
  - the serialized answer block emits the PRD §5 fields **in the required order**
  - the exact seven label strings, in canonical order
  - a question set **without** a gold block validates successfully
- [x] `python -m ats.answer --help` and `python -m ats.eval --help` both exit 0
- [x] `pip install -e .` succeeds from a clean environment on both members' machines — verified on Member A's machine 2026-09-10 (fresh `.venv`, Windows Store Python 3.11.9): `pip install -e ".[dev]"` clean, `pytest` 50/50 passing, both CLI `--help` entry points exit 0
- [x] Target device recorded in §0 of this document
- [x] `docs/CITATIONS.md` exists with its first entry, establishing the append-as-you-go habit from commit one

### Artifacts produced

Four frozen schemas · both CLI signatures · the `evaluate()` signature · working dev environment.

---

## Phase 1 — Two Independent Vertical Slices

**Type:** Parallel. Neither member depends on the other's output during this phase.

**Objective:** each member builds their end of the pipeline against a *fake* of the other end. By the end of this phase both halves run end to end in isolation.

### Member A — Signals & Systems

- **1A.1** `scripts/fetch_data.py` — download and stage ExtraSensory. No raw data committed (PRD §3.3).
- **1A.2** Ingest and parse; restrict to accelerometer + gyroscope; map ExtraSensory labels onto the seven canonical classes.
- **1A.3** **Resample every stream to exactly 25 Hz** (PRD §3.2). Handle irregular/jittered timestamps.
- **1A.4** Explicit **gap policy**: gaps beyond a stated maximum are *marked, not interpolated*; short gaps are interpolated and reflected in the window's `coverage` value. Document the threshold and the reasoning.
- **1A.5** Windowing. **Choose window length and hop, then freeze and tell B** — the hop determines the temporal resolution of every interval B can produce.
- **1A.6** **Subject-wise** train/val/test splits, persisted to `splits/*.json` and committed. Splitting by window instead of by subject leaks the same person across train and test and inflates every number we report.
- **1A.7** Label-ambiguity policy: ExtraSensory labels are self-reported and can overlap (PRD §3.2). Define and document the single-label collapse rule.
- **1A.8** **`ats/oracle.py` — the unblocking deliverable.** Emits a schema-valid `window_track` built from *ground-truth labels* instead of a model, with flags:
  ```
  --label-noise p        inject p fraction of wrong labels
  --soften-confidence    emit realistic probability spreads rather than one-hot
  --drop-windows p       simulate missing windows
  ```
  > This single file is what removes B's dependency on the classifier. B builds the entire reasoning and evaluation stack against it and is never blocked. The noise flags matter: a reasoning layer tested only against a *perfect* oracle will break the moment a real classifier is attached.

### Member B — Reasoning & Evaluation

- **1B.1** `ats/aggregate.py` — window track → timeline of `(activity, t_start, t_end, mean_confidence)` intervals. **Temporal smoothing and segmentation live here**, not in A's classifier (see §"Ownership boundary notes").
- **1B.2** `ats/serialize.py` — answer object → the exact PRD §5 text block, plus `--format jsonl` for machine scoring.
- **1B.3** `ats/eval/metrics.py` — the complete PRD §7.3 metric library:
  - categorical: accuracy, macro-F1, balanced accuracy
  - binary verification: precision, recall, F1 on the positive class, **plus specificity**
  - numeric: tolerance accuracy, MAE, MAPE
  - temporal: interval IoU, temporal precision/recall/F1, multi-interval matching
  - combined **grounded accuracy**: answer correct **AND** IoU ≥ τ **AND** modality/channels match
- **1B.4** `data/questions_dev.json` — hand-authored dev question set, **≥12 questions per tier** across ≥3 dev subjects, with gold. Must include deliberate edge cases:
  - a question about an activity that never occurs in the recording
  - a question whose answer interval lands inside a data gap
  - a comparison question that is a genuine tie
  - a duration question spanning multiple non-contiguous intervals

  > PRD §7.1 warns the graded set will include difficult and edge-case questions. Authoring our own edge cases now is the only way to find out that our system returns `None` on them.

  > **Status (revised 2026-09-11):** 53 questions over three **synthetic** subjects (12/17/12/12 across tiers 1–4), one file per subject in `data/questions_dev/` so each question set pairs with exactly one recording, as it will at evaluation time. The fixtures were regenerated in Phase 2 to mirror ExtraSensory's real structure (one 22 s burst per labeled minute; gaps are missing minutes) after the minute-attribution decision in §0. All four edge cases are present and asserted by `tests/test_pipeline.py`. Gold is derived from the declared ground truth in `scripts/make_dev_fixture.py`, never from the reasoning code, so using this set to test that code is not circular. **Real subjects, 2026-09-11:** 34 questions over two real ExtraSensory subjects in `data/questions_dev_v2/`, generated by `scripts/make_real_dev_questions.py` (see the Phase 3 caveats). Both are training-split subjects, so the "≥3 dev subjects" requirement is still not fully met: at least one more real subject is needed, ideally all from the validation split.

### Exit criteria

**Member A**
- [x] `pytest tests/test_preprocess.py` passes, asserting on synthetic signals:
  - jittered input timestamps resample to **exactly** 25 Hz, with zero duplicate or backward timestamps
  - a known 1.5 Hz sinusoid survives resampling within a stated tolerance (proves we did not destroy gait-band content)
  - an injected gap produces `coverage < 1` on **exactly** the overlapping windows and no others
- [x] `pytest tests/test_splits.py` asserts **zero subject overlap** across train/val/test — verified both on synthetic subject IDs and on the real 60-subject split in `splits/subject_splits.json`
- [x] `python -m ats.oracle --subject <id> --out track.jsonl` produces a file that validates against `window_track.schema.json` — verified both against a controlled synthetic fixture (`tests/test_oracle.py`) and end-to-end against a real ExtraSensory subject (21,505 windows, 100% schema-valid; see `tests/fixtures/track_subj_real_00EABED2.jsonl`)

**Member B**
- [x] `python -m ats.answer --questions data/questions_dev/<subject>.json --track <track> --out ans.txt` produces **100% schema-valid output for 100% of questions**. Content correctness is *not* gated yet — well-formedness is. Gated automatically by `tests/test_pipeline.py`.
- [x] `python -m ats.eval` runs to completion and emits a metrics dict
- [x] `pytest tests/test_metrics.py` — every metric checked against a **hand-computed** fixture: a two-interval IoU worked out by hand, a macro-F1 on a toy 3-class confusion matrix, an MAPE on known values

> **Re-run against the real oracle track, 2026-09-11.** All three criteria above were first validated against `tests/fixtures/track_subj_synth_*.jsonl` (synthetic). Now that Member A's `ats/oracle.py` and `tests/fixtures/track_subj_real_00EABED2.jsonl` exist, they were re-run against the real track: `python -m ats.answer` produced 100% schema-valid output for all 52 dev questions (0 missing predictions), `python -m ats.eval` ran to completion, and the frozen `DURATION_ABS_TOL_S = 4.0` (2 × the 2.0s hop) was picked up automatically. This is a **well-formedness** check only — the ~10-12% accuracy that run produced is expected and meaningless as a result, since the dev question set's gold answers describe the *synthetic* subjects, not `00EABED2`, and answer content is still just the Phase 1 dominant-activity baseline. No content-correctness claim is made here.

### Artifacts crossing the boundary

- **A → B:** `ats/oracle.py` + one committed sample track (`tests/fixtures/track_subj_real_00EABED2.jsonl`, regenerable via `scripts/make_sample_track.py`) + the frozen window/hop numbers (4.0s / 2.0s) — **delivered 2026-09-10**
- **B → A:** `data/questions_dev/` (one question set per dev subject) + the importable `evaluate()` + `ats.eval.dev.run_dev_eval()`. The Phase 1 single-file set was superseded on 2026-09-11, before anything consumed it, when the minute-attribution decision changed what a gold interval means. **Frozen from that date** — later additions go to `data/questions_dev_v2/` so A's robustness and Pareto curves stay comparable across the project

---

## Phase 2 — Real Recognition, Real Reasoning

**Type:** Parallel. Still decoupled — A works against real data, B still works against the oracle.

### Member A — Signals & Systems

- **2A.1** Feature extraction: time-domain and frequency-domain features per window (magnitude statistics, cadence/dominant frequency, spectral energy, jerk, correlation between axes).
- **2A.2** Baseline classifier — engineered features + gradient boosting, **or** a compact 1D-CNN. Pick one and justify it in the report. **Bias the choice toward something that quantizes cleanly**, since Phase 5 depends on it.
- **2A.3** Explicit class-imbalance handling (class weights or balanced sampling), justified in writing — PRD §3.2 flags the sedentary skew as part of the problem, not an excuse.
- **2A.4** Serialize `models/full/` and generate a `window_track` for every dev subject.
- **2A.5** `scripts/make_fig2.py` — confusion matrix over the 7 classes plus per-class precision/recall/F1 (**Figure 2**).

### Member B — Reasoning & Evaluation

- **2B.1** **Question routing** — map question text to one of a closed set of typed operators:
  `identify` · `verify` · `duration` · `count` · `when/onset` · `compare` · `ground` · `open_world`.
  **Implement rule/keyword routing first.** This is the fallback that keeps the system working when the SLM misparses, and it is the baseline the SLM must beat in Phase 4.
- **2B.2** Deterministic reasoning operators over the timeline. **Every numeric answer is computed in Python from intervals — never emitted by a model.** Durations sum interval lengths; counts count intervals; comparisons compare totals.
- **2B.3** **Evidence attributor** — for any answer, return supporting intervals + modality + channel(s) + the `feature_summary` values that justify them. This is what fills PRD §5's Evidence block with something real.
- **2B.4** **Answer validator** — rejects any answer that:
  - cites an interval not present in the timeline, or
  - states a number that disagrees with recomputation from the timeline, or
  - carries `N/A` evidence on a tier-3 or tier-4 question

  > This is the architectural enforcement of PRD §1.3 ("an answer produced by language reasoning alone does not meet the requirement") and directly protects the 20% evidence-grounding weight. It is a hard gate in the code path, not a lint.

  > **Status:** implemented in `ats/routing.py`, `ats/operators.py`, `ats/evidence.py`, and `ats/validator.py`, with shared vocabulary in `ats/vocab.py`. Two design calls worth knowing. (1) A rejected answer is replaced by an explicit abstention that itself passes validation, and the rejection is reported, so nothing unvalidated is ever written. (2) An abstention (`N/A`) is the one answer allowed without evidence at tiers 3–4, because it makes no claim; any other tier-3/4 answer must cite evidence, and negative findings ("never happened") cite every observed interval as the evidence examined. Questions the operators cannot answer faithfully (time-restricted durations, threshold questions, unrecognised open-world phrasing) abstain rather than answer a different question.

### Exit criteria

**Member A**
- [x] Window-level macro-F1 on **held-out subjects** beats two baselines computed in this same phase: (a) majority class, (b) logistic regression on per-window mean/std. The bar is *beat both*, with the margin recorded in the commit message — an absolute number invented in advance would be meaningless.
  > **Measured 2026-09-11**, retrained after fixing docs/bug.md issue 1 (accelerometer units), on the real 60-subject sample (36 train / 12 val / 12 test, `splits/subject_splits.json`), via `scripts/train_cnn.py` on Kaggle: majority-class macro-F1 **0.0792**, logistic regression (mean/std) macro-F1 **0.2251**, `ActivityCNN` macro-F1 **0.2893**. Beats (a) by **+0.2101** and (b) by **+0.0642**. (Pre-fix numbers, superseded: LR 0.1555, CNN 0.2749 — the units bug directly corrupted the LR baseline's input, which is why it moved more than the CNN's did.) Full numbers in `models/full/phase2_results.json`, analysis in `docs/results_recognition.md`.
- [x] Figure 2 generated with per-class precision/recall/F1 table alongside — `results/fig2_confusion_matrix.png` + `results/fig2_per_class_prf.csv`, via `scripts/make_fig2.py`.
- [x] Confusions are named in writing, especially sitting vs standing-still and walking vs running (PRD §7.4.2 calls these out specifically) — see `docs/results_recognition.md`: walking-vs-running confused mostly one-directionally post-fix (6.0% walking->running, 23.5% running->walking); sitting-vs-lying is the single largest confusion in this sample and grew after the fix (43.8%, up from 29.5% pre-fix, larger than sitting-vs-standing-still here); STANDING_STILL (precision 0.036) and RUNNING (precision 0.004, 68.7x rarer than the majority class) both explained as structural failures — signal ambiguity and data scarcity respectively — not training-recipe bugs.

**Member B** *(measured against the **oracle** track)*
- [x] Tier-1 and tier-2 accuracy **≥ 0.95** — measured **1.000** (n=12) and **1.000** (n=17)
- [x] Grounded accuracy at IoU 0.5 **≥ 0.90** — measured **1.000** (n=26)
  > With a perfect classifier the reasoning layer has no excuse. Anything below these numbers is a bug in B's code, and testing against the oracle isolates that cleanly from recognition error.
- [x] `pytest tests/test_validator.py` — a deliberately fabricated answer (interval absent from the timeline; duration off by 30 s) is **rejected**, and the pipeline is shown never to emit a rejected answer
- [x] Reasoning still degrades gracefully with `--label-noise 0.1` on the oracle (no crashes, no empty answers) — measured: 0 crashes, 0 empty answers, 0 missing predictions. Before the per-burst vote (see §0), accuracy by tier was 0.917 / 0.941 / 0.917 / 0.833 and grounded 0.962; after it, isolated window flips are absorbed entirely and every tier scores 1.000. Because that noise no longer stresses anything, whole-burst corruption (`--burst-noise 0.1`, the correlated error a real classifier makes) is also measured: tiers 1.000 / 0.706 / 0.667 / 0.833, grounded 0.692, still with 0 crashes, empty or missing answers (`results/phase2_dev_eval_burst10.json`)

> **What these numbers do and do not show.** Measured 2026-09-11 by `python scripts/run_dev_eval.py` (with and without `--label-noise 0.1 --seed 0`; outputs in `results/phase2_dev_eval*.json`) and gated permanently by `tests/test_pipeline.py`. They were measured against the three synthetic oracle-style fixture subjects, not the real oracle: the real oracle needs the raw dataset, which is not on Member B's machine, and there are no gold questions for real subjects yet. A perfect score from a perfect classifier is expected by construction. It shows that routing, operators, and validator compute the right answer from a correct timeline, and says nothing about accuracy on real recordings. The 2 abstentions in the clean run are the two data-gap questions, where `N/A` is the correct answer. The first real numbers come in Phase 3.

---

## Phase 3 — The Swap

**Type:** Joint. A single, scheduled integration event.

**Objective:** replace the oracle track with A's real classifier output. Because both sides have spoken the frozen `window_track` schema since Phase 0, this is a **config change, not a merge**.

### Joint tasks

- **3.1** Run end to end on dev subjects with `--model models/full`. Produce the first honest QA numbers.
- **3.2** Build `scripts/oracle_delta.py` — runs the **identical** question set through both the oracle track and the real track and diffs the results, attributing every regression to one of:
  - **recognition error** (real track wrong where oracle was right)
  - **aggregation error** (both tracks agree, intervals still wrong)
  - **routing error** (wrong operator selected regardless of track)

  > This is the single most valuable debugging instrument in the project. Without it, a wrong answer is just a wrong answer and both members argue about whose fault it is. With it, every failure has an owner within seconds.

  > **Status (2026-09-11): built ahead of the swap.** `scripts/oracle_delta.py` (logic in `ats/eval/delta.py`, tests in `tests/test_oracle_delta.py`). "Aggregation error" is reported as **reasoning** (aggregation or operator, both Member B), and an answer wrong on the oracle but right on the real track is listed separately as a *masked bug*. "Right" includes grounding wherever the gold cites evidence, so evidence-only regressions are attributed too. Tracks are found by name, `<dir>/track_<subject>.jsonl`, one per question set in `data/questions_dev/`. **For Member A (2A.4):** write each dev subject's oracle and real tracks to `results/raw/tracks/oracle/` and `results/raw/tracks/full/` (both gitignored), then run `python scripts/oracle_delta.py --oracle-dir results/raw/tracks/oracle --real-dir results/raw/tracks/full`. Before a model exists, `--simulate-burst-noise P` (whole bursts misclassified, the realistic case) or `--simulate-label-noise P` (isolated window flips, which the burst vote now absorbs entirely) does a dry run; either is labelled SIMULATED throughout and written to `results/oracle_delta_simulated.*`. Only a real-track run counts toward the exit criteria below.
- **3.3** Triage the delta table; each member leaves with an ordered fix list **for their own layer only**.

### Exit criteria

- [x] `python -m ats.answer` on a dev recording with the real model produces **100% schema-valid** answers — measured 2026-09-11 on both real subjects' real-model tracks (`tests/fixtures/real_model_tracks/`): 17/17 and 17/17 answers schema-valid, 0 withheld by the validator.
- [x] `scripts/oracle_delta.py` emits a per-question-type table attributing every loss to a layer — measured 2026-09-11 on 34 questions over two real subjects (`data/questions_dev_v2/`); `results/oracle_delta.md` (labelled time only, the default) and `results/oracle_delta_all_windows.md` (every recorded window) give identical per-type numbers:

  | Question type | n | Right (oracle) | Right (real) | Lost: recognition (A) | Lost: routing (B) | Lost: reasoning (B) |
  |---|---|---|---|---|---|---|
  | identification | 4 | 4 | 2 | 2 | 0 | 0 |
  | verification | 4 | 4 | 4 | 0 | 0 | 0 |
  | duration | 5 | 5 | 0 | 5 | 0 | 0 |
  | count | 2 | 2 | 0 | 2 | 0 | 0 |
  | comparison | 3 | 3 | 1 | 2 | 0 | 0 |
  | grounding | 8 | 8 | 2 | 6 | 0 | 0 |
  | open_world | 8 | 8 | 6 | 2 | 0 | 0 |
  | **overall** | **34** | **34** | **15** | **19** | **0** | **0** |

  No masked bugs, and nothing withheld on either track.
- [ ] Delta table committed to `results/` and reviewed by both members — committed and reviewed by Member B; **Member A still to review**.
- [ ] Both members have a written, ordered fix list — the table's fix lists are in `results/oracle_delta.md`: 19 recognition items for Member A, still to be put in priority order, and none for Member B. Member B's open items come from elsewhere: the five rule-router gaps found in the Phase 4 ablation, and whether onset questions should cite only the onset minute.

> **Read these numbers with four caveats.**
> 1. **Both subjects are in the training split** (`splits/subject_splits.json`), so this is the model on people it was trained on, and held-out subjects will do worse. Even so, window-level accuracy on labelled time is only 0.503 (subj_real_a) and 0.673 (subj_real_b), with 47% and 32% of true sitting windows predicted as lying. Repeat on validation-split subjects before any of this is reported as performance.
> 2. **Compared over labelled time by default.** The oracle skips recorded minutes nobody labelled (4,232 real-model windows across the two subjects). A prediction there has no truth to be judged against, so the delta sets it aside rather than charge it to recognition; `--include-unlabelled` scores every recorded window instead.
> 3. **The oracle is right by construction.** `scripts/make_real_dev_questions.py` derives each gold bout from the ground-truth labels with an implementation of the §0 rules that is independent of `ats/aggregate.py`, and refuses to write unless the two agree exactly (they do, on both subjects). That checks the aggregation on real data, but it also means this table cannot surface an aggregation error: every oracle-vs-real difference is recognition's by design.
> 4. **The bout counts in `docs/phase3_handoff.md` predate the drift fix** (§0, interval semantics). Bridging sub-minute timing drift took subj_real_b from 3,151 bouts to 325 and its longest lying stretch from 624 s to 20,904 s; the gold in `data/questions_dev_v2/` uses the corrected semantics. The data-gap question now sits only in stretches with no recorded windows at all, after a first draft for subj_real_b landed in a stretch that was recorded but unlabelled.

---

## Phase 4 — Language Interface and Open-World Reasoning

**Type:** Parallel. B-heavy on features, A-heavy on hardening.

### Member B — Reasoning & Evaluation

- **4B.1** SLM integration. Candidates per PRD §8.1: Qwen2.5-1.5B-Instruct, Llama-3.2-1B/3B, Ministral-3. **Prefer the smallest model that parses reliably** — latency counts against us in PRD §6.1, and the SLM dominates per-query cost.
- **4B.2** Constrain the SLM to **exactly two roles**:
  1. **Parser:** natural language → a typed operator call, validated against a JSON schema, with the Phase 2 rule router as automatic fallback on parse failure.
  2. **Phraser:** template-anchored explanation text over *already-computed* evidence.

  The SLM never produces a number, an interval, or a verdict. Those come from the deterministic operators. This is how PRD §8's hard constraint — "the language it produces must be tied to evidence the earlier layers found, not generated freely" — becomes a property of the architecture rather than a hope.
- **4B.3** **Task 4 open-world path.** Map free-form questions to *signal-property predicates* computed from `feature_summary` — sustained-low-variance, cyclic-cadence-without-impact-spikes, sustained-periodic-gyro-oscillation, and so on. This is what lets the system argue about a behavior it was never trained to name (PRD §4.4) from measurements rather than from the 7-label vocabulary.

  > **Status (2026-09-11): stillness only, and not yet validated across phones.** `ats/signal.py` judges a window *still* from the signal alone: accelerometer-magnitude standard deviation at most 0.022 m/s² and gyroscope energy at most 0.005, the rounded-up 95th percentiles of lying-down windows on subj_real_a, the calibration subject (re-derived by `tests/test_signal.py`). Stillness backs three open-world behaviours. A rest question that names no posture is answered from the longest run of mostly still signal, so the classifier's sitting-versus-lying confusion does not matter. And "wheeled" and "strenuous" claims must rest on intervals whose signal is mostly moving. subj_real_a has no running or bicycling, so no movement signature is calibrated and none is claimed.
  >
  > **The check on subj_real_b, which was kept aside for it, failed for a reason upstream of this layer.** Its median acceleration magnitude is 94.75 m/s², against 9.78 on subj_real_a: about 9.7 times gravity, so none of its 41,000 windows can look still (`results/stillness_calibration.json`). Stillness is therefore only judged on recordings whose median magnitude lies within 8–12 m/s², a band set from physics rather than from either subject; elsewhere the rest question abstains and movement labels stand unvetoed. On the dev set the checks changed no answer: subj_real_a's spurious bicycling minutes have a moving signal, so the veto does not catch them.
  >
  > **For Member A (full report, evidence and fix checklist in [bug.md](bug.md)):** the 9.7× factor looks like a units problem in `ats/ingest.py`, plausibly subjects whose phones already record m/s² being converted from g a second time. If so, it also feeds inconsistent input scales into the classifier's training data, and the explanations for affected subjects quote impossible accelerations. Once it is fixed, re-run `scripts/calibrate_stillness.py`; subj_real_b is still the check.
- **4B.4** LLM-judge rubric for explanation quality: 1–5 scale over the three PRD §7.3.5 sub-criteria (cites real signal features / features support the conclusion / conclusion plausible).
- **4B.5** **Human double-rating of a ≥20-explanation sample** to produce the inter-rater agreement number PRD §7.3.5 asks for. Both members rate independently, then compare.
- **4B.6** Instrument the interface layer using A's profiler (from 4A.2).

### Member A — Signals & Systems

- **4A.1** Work the recognition fix list from the Phase 3 delta table; focus on sedentary-class disambiguation, which is where the confusion matrix will be worst.
- **4A.2** `ats/profile.py` — the cost-measurement harness: parameter count, on-disk MB, peak RSS, per-query latency (p50/p95), CPU utilization, energy estimate. Emits `cost_report.schema.json`, naming the target device.
  > A owns the harness; **B instruments the interface layer with it**. PRD §6.1's "time taken to answer a single query" includes SLM inference, so A cannot measure it alone — and two independently written profilers would produce two incomparable numbers.

### Exit criteria

- [x] **Ablation committed:** SLM-routed vs rule-routed accuracy on the frozen dev set. If the SLM does not beat the rules, **keep rules as the default path and say so in the report** — a measured negative result is a defensible design finding, not a failure. — **Measured 2026-09-11** by `scripts/run_router_ablation.py` (`results/phase4_router_ablation.json`): Qwen2.5-0.5B-Instruct, few-shot prompt, greedy decoding, closed-schema validation with the rule router as fallback.

  | | Rules | SLM (+ rule fallback) |
  |---|---|---|
  | Dev-set routing accuracy (n=53) | 1.000 | 0.868 |
  | Dev-set QA accuracy, macro (n=53) | 1.000 | 0.819 |
  | Held-out routing, questions from the brief (n=6) | 0.83 | 0.67 |
  | Held-out routing, written by Member B (n=13) | 0.69 | 0.46 |
  | Routing latency p50 / p95 (informal, target laptop) | 0.3 / 0.4 ms | 5.6 / 7.5 s |

  > **Decision: rules stay the default; the SLM remains available behind `--router slm`.** It beat the rules on nothing, fell back to them on 30 of 125 calls (24%), and costs about 5 s per question. Its commonest failure is collapsing to `open_world` with an operator name ("onset", "ground") in the predicate field. Caveats for the report: the dev set was templated alongside the rules, so their 1.000 there is optimistic, which is why the held-out set exists; 6 questions from the brief is a tiny sample; and this measures one prompt design, not the ceiling of small models. The held-out run also exposed five rule gaps, all phrasings outside the templates: "is her walking time increasing week to week" routed to a yes/no check (a confident answer to an unsupported question), plus "in total, how much sitting", "count the separate running episodes", "lying down for ages" and "mostly sitting around or up and about". Fixing the rules against those same questions would stop them being held out, so a fresh blind set, written by someone who has not read `ats/routing.py`, is needed before final numbers.
- [x] `pytest tests/test_no_ungrounded_output.py` — asserts every tier-3 and tier-4 answer carries a non-`N/A` interval that **exists in the timeline**. Zero validator rejections escape to output. — Passes (2026-09-11) on every dev question against every track available: the three synthetic oracle tracks, the same with 20% of bursts mislabelled, and both real subjects on their oracle and real-model tracks. Every emitted answer passes the validator, every tier-3/4 claim cites evidence, and the operators never needed overruling. An explicit abstention (`N/A`) is the one tier-3/4 answer allowed without evidence, because it makes no claim (see 2B.4).
- [ ] `python -m ats.profile --config full` emits a schema-valid cost report naming the target device
- [ ] Mean rubric score computed, with an inter-rater agreement figure from the ≥20-explanation sample
- [ ] Open-world questions about behaviors outside the 7 classes return an argued answer, not a nearest-label guess

---

## Phase 5 — Compression, Efficiency, and Robustness

**Type:** A-owned, with a frozen contribution from B. **Required** — this is the PRD §6.3 extra credit and it is not optional in this plan.

### Member A — Signals & Systems

- **5A.1** Produce **≥3 compressed configurations** beyond `full`: 8-bit quantized, pruned, distilled. Each writes `models/<cfg>/` plus its own cost report.
- **5A.2** Degradation injectors for the robustness curve (PRD §7.4.5): additive sensor noise at stated SNR levels, sample dropping at stated percentages, and decimation below 25 Hz followed by upsampling.
- **5A.3** `scripts/sweep.py` — runs every config × every degradation level, calling `evaluate()` **directly as a function** (not via subprocess). Writes `results/pareto.csv` and `results/robustness.csv`.
- **5A.4** `scripts/make_fig4.py` and `scripts/make_fig5.py` — accuracy-vs-overhead scatter with the Pareto frontier drawn, and the robustness curve.

### Member B — Reasoning & Evaluation

- **5B.1** **Freeze the question set and all metric definitions for the duration of the sweep.** A curve computed against a moving question set compares nothing.
- **5B.2** Pareto/non-dominated-set helper in `ats/eval/`, consumed by A's figure script.
  > **Done 2026-09-11:** `ats.eval.pareto.pareto_frontier(points, cost, value)` returns the non-dominated configurations sorted by rising cost (lower cost and higher value are better); tested against a hand-worked fixture in `tests/test_curves.py`.
- **5B.3** Produce **Figure 1** (accuracy by question type) and **Figure 3** (accuracy vs strictness) from the `full` config.
  > **Scripts done, numbers still draft (2026-09-11).** `scripts/make_fig1.py` and `scripts/make_fig3.py` answer every question set against the oracle and real-model tracks and write each figure with a CSV (the table view) and a caption file; Figure 1's caption states each group's correctness rule, as PRD §7.4.1 requires. The data is computed in `ats/eval/curves.py` and tested separately from the plotting. The palette (reference slots 1 and 2) passes the dataviz validator for light mode. Current values on the two real subjects: trained model 42% overall (macro) against 100% from oracle labels; 35% of cited intervals accepted at the pre-registered IoU of 0.5; no duration within 10%. **Do not put these numbers in the report yet:** both subjects are in the training split, and the accelerometer units issue (see 4B.3) is unresolved. Re-run both scripts once the validation-split subjects and the units fix land.

### Exit criteria

- [ ] `results/pareto.csv` has **≥4 rows** (full + 3 compressed), each carrying accuracy **and ≥2 cost axes**
- [ ] **≥1 compressed point is non-dominated** and shows a large cost reduction for a small accuracy reduction — this is precisely what PRD §6.3 rewards. If no such point exists, **report the negative result with the curve** rather than hiding it; a shown-and-explained flat tradeoff scores better than a missing figure.
- [ ] `results/robustness.csv` has **≥4 degradation levels** on at least one axis
- [ ] `scripts/make_all_figures.py` regenerates **all five** figures from `results/*.csv` with **zero manual steps**
- [ ] Every figure caption states the correctness rule used (mandatory for Figure 1 per PRD §7.4.1) and the thresholds chosen

---

## Phase 6 — Evaluation-Time Hardening, Deliverables, and Demo

**Type:** Joint, with split writing duties.

**Objective:** the graded recordings and questions are undisclosed until evaluation time (PRD §7.1). This phase simulates that as closely as we can.

### The Cold-Start Drill *(the most important task in this phase)*

Each member:
1. Clones the repo into a **fresh directory and fresh environment**
2. Follows the README section **written by the other member**
3. Runs `python -m ats.answer` on a subject **never used during development**
4. Using a question set **the other member wrote that morning**, unseen until that moment

Every failure here is a failure that would otherwise have happened live on grading day. Commit the failure log; fix every item; re-run.

### Joint tasks

- **6.1** Cold-start drill, both directions.
- **6.2** Architecture diagram for the report (A draws the signal path, B the reasoning path, one shared figure).
- **6.3** Final consistency pass: the timestamp convention is identical across every answer, every figure axis, and every table in the report.
- **6.4** Rehearse the demo end to end **on a machine other than the development machine**.

### Member A

- **6A.1** README: environment setup, data fetch and preparation, model artifacts, and the exact commands that reproduce our reported numbers (PRD §9.2 requires the teaching team be able to rerun us).
- **6A.2** Reproduce Figures 2, 4, 5 from a clean checkout.

### Member B

- **6B.1** README: the run/eval section — fresh recording + question set → PRD §5 output.
- **6B.2** Reproduce Figures 1 and 3 from a clean checkout.
- **6B.3** Fold `docs/CITATIONS.md` into the report's references so every external resource is cited at its point of use (PRD §9.3).

### Exit criteria

- [ ] Cold-start drill passes **in both directions**, from a clean clone, with **zero undocumented steps**; the failure log is committed
- [ ] Commit history is spread across many distinct days for **both** authors — check per author:
  ```bash
  git log --author="<name>" --format=%ad --date=short | sort -u | wc -l
  ```
  PRD §9.2 states the history is part of how effort is judged; a single end-of-project upload is explicitly against the rules
- [ ] Report is **10–12 pages** and contains: problem framing, design rationale, implementation, results across all four tasks, all five figures with compliant captions, threshold justifications, the accuracy-vs-overhead analysis, and the **per-member contribution statement**
- [ ] No raw dataset in the repository; the fetch/prepare path works from scratch
- [ ] Demo rehearsed on a second machine

---

## Figure Ownership

| Figure | PRD ref | Owner | Note |
|---|---|---|---|
| 1 — Accuracy by question type | §7.4.1 | **B** | Caption **must** state the correctness rule per group, since the rule differs across groups |
| 2 — Activity confusion matrix + per-class P/R/F1 | §7.4.2 | **A** | Window-level, recognition backbone |
| 3 — Accuracy vs strictness (IoU 0.1–0.9; tolerance sweep) | §7.4.3 | **B** | It is B's metric definitions being swept |
| 4 — Accuracy vs overhead + Pareto frontier | §7.4.4 | **A** | A produces the points; B supplies `evaluate()` and the frontier helper |
| 5 — Robustness curve | §7.4.5 | **A** | A owns the injectors; question set frozen by B |

## Report Section Ownership

| Section | Owner |
|---|---|
| Problem framing and motivation | B |
| System design overview + architecture diagram | Joint — A writes the signal path, B the reasoning path |
| Implementation: preprocessing, windowing, splits, recognition | A |
| Implementation: aggregation, routing, SLM interface, serializer | B |
| Results: recognition, Figure 2 | A |
| Results: QA accuracy, metric definitions, threshold justification, rubric + inter-rater agreement, Figures 1 & 3 | B |
| Efficiency, target-device disclosure, compression, Pareto, robustness, Figures 4 & 5 | A |
| Limitations and future work | Joint |
| Per-member contribution statement | Each writes their own paragraph; both countersign |

---

## Contribution Statement Scaffold

Drafted now so the final statement summarises the plan rather than reconstructing it from memory.

> **Member A (Signals & Systems).** Owned the data path end to end: ExtraSensory acquisition and parsing, 25 Hz resampling with an explicit gap policy, windowing, and subject-wise splitting. Built and trained the seven-class recognition backbone including class-imbalance handling, and produced the confusion-matrix analysis. Owned all efficiency work: the profiling harness, three compressed model configurations, the accuracy-versus-overhead Pareto analysis, and the robustness experiments. Wrote the setup/reproduction half of the README and the recognition, efficiency, and robustness sections of the report.

> **Member B (Reasoning & Evaluation).** Owned the reasoning path end to end: aggregation of per-window predictions into an activity timeline, temporal segmentation, evidence attribution, and the deterministic reasoning operators behind every numeric and temporal answer. Built the question router and the constrained small-language-model interface, together with the validator that prevents any ungrounded answer from being emitted. Owned the entire evaluation harness — every correctness rule in the metric library, the dev question set, the explanation rubric and inter-rater study — and produced the accuracy-by-question-type and accuracy-versus-strictness analyses. Wrote the run/evaluate half of the README and the framing, reasoning, and results sections of the report.

---

## Ownership Boundary Notes

Three boundaries are drawn differently from the obvious layer split. Each is deliberate.

1. **Temporal smoothing and segmentation belong to B, not to A's classifier.** They operate on the prediction *track* and directly set the interval boundaries that grounding IoU is scored against. They must sit with whoever is optimizing that metric, or B is held accountable for a number A controls.

2. **A owns the profiling harness; B instruments the interface layer with it.** PRD §6.1's per-query latency includes SLM inference, so A cannot measure the full number alone. Two independently written profilers would produce two incomparable numbers and undermine the whole efficiency section.

3. **Robustness splits at the injector/metric line.** A owns the degradation injectors and runs the sweep; B owns the frozen question set and `evaluate()`; A produces the figure. This keeps the measurement instrument in the hands of the person not being measured.

---

## Risk Register

| # | Risk | Mitigation, and the phase that carries it |
|---|---|---|
| 1 | **Integration bang** — the two halves meet late and nothing fits | Schemas frozen in Phase 0; Phase 3 is a *scheduled swap*, not a discovery. `oracle_delta.py` keeps layer attribution possible afterwards |
| 2 | **B blocked waiting on A's classifier** | `ats/oracle.py` ships in Phase 1. B's Phase 2 exit gate is *defined against the oracle*, so B reaches "done" with no real model in existence |
| 3 | **A blocked waiting on B's harness** | A's Phase 2 gate is window-level macro-F1, entirely self-contained. `evaluate()` is importable and frozen before A needs it in Phase 5 |
| 4 | **SLM emits ungrounded answers** — the single largest grading risk (20% evidence grounding, plus the PRD §1.3 disqualifier) | SLM confined to parser + phraser roles only; deterministic operators produce every number; Phase 2 validator rejects fabrications; rule router as fallback; a test fails the build if any tier-3/4 answer cites a non-existent interval |
| 5 | **Overfitting to our own question set** | Phase 6 cold-start drill on an unseen subject with questions written by the other member; question set frozen before the Phase 5 sweeps |
| 6 | **Thin or bursty commit history** (explicitly against PRD §9.2) | Repo exists from Phase 0; every phase produces committable artifacts on *both* sides; Phase 6 checks distinct commit days per author |
| 7 | **Extra credit deferred, then dropped** | Phase 5 is required with a hard `pareto.csv` row-count gate; A biases the Phase 2 architecture choice toward something that quantizes cleanly |
| 8 | **Threshold-shopping after seeing results** | IoU τ = 0.5 and the duration tolerance are pre-registered in §0 *before* any results exist; the full sweep is reported as Figure 3 so the choice is visible rather than load-bearing |
| 9 | **Class imbalance flatters our headline number** | Overall QA accuracy is macro-averaged across question types (PRD §7.3); macro-F1 and balanced accuracy reported alongside plain accuracy throughout |

---

## Traceability: PRD Requirement → Phase → Owner

| PRD requirement | PRD § | Phase | Owner |
|---|---|---|---|
| Scenario / motivation | §1.1 | 6 (report) | B |
| No language-only answers | §1.3, §8 | 2 (validator), 4 (constrained SLM) | B |
| Dataset, modalities, 7 classes | §3.1 | 1 | A |
| 25 Hz resampling | §3.2 | 1 | A |
| Handle noise / gaps / imbalance | §3.2 | 1, 2, 5 | A |
| Don't commit raw dataset | §3.3, §9.2 | 0, 1 | Joint |
| Task 1 — identification | §4.1 | 2 | B |
| Task 2 — temporal / quantitative | §4.2 | 2 | B |
| Task 3 — evidence grounding | §4.3 | 2 | B |
| Task 4 — open-world reasoning | §4.4 | 4 | B |
| Output format fields and order | §5 | 0 (schema), 1 (serializer) | B |
| Timestamp convention stated | §5.1 | 0 | Joint |
| Resource cost reporting | §6.1 | 4 | A (B instruments interface) |
| Target device disclosure | §6.2 | 0 | Joint |
| Edge extra credit + Pareto proof | §6.3 | 5 | A |
| Generalize, don't overfit | §7.1 | 6 (cold-start drill) | Joint |
| Grading weights (design 25 / correctness 20 / accuracy 35 / grounding 20 / +10) | §7.2 | all — sets prioritisation | Joint |
| Correctness rules per answer type | §7.3 | 1 | B |
| Macro-averaged overall QA accuracy | §7.3 | 1 | B |
| Open-world rubric + inter-rater agreement | §7.3.5 | 4 | B |
| Figure 1 — accuracy by question type | §7.4.1 | 5 | B |
| Figure 2 — confusion matrix | §7.4.2 | 2 | A |
| Figure 3 — accuracy vs strictness | §7.4.3 | 5 | B |
| Figure 4 — accuracy vs overhead | §7.4.4 | 5 | A |
| Figure 5 — robustness curve | §7.4.5 | 5 | A |
| Layered architecture | §8 | 0–4 | Joint |
| SLM choice justified | §8.1 | 4 | B |
| Related readings reviewed | §8.2 | 0 | Joint |
| Groups of 2–3, contribution statement | §9.1 | 6 | Joint |
| Incremental commit history | §9.2 | all | Joint |
| README + reproducible setup | §9.2 | 6 | A (setup) + B (run/eval) |
| External resources cited at point of use | §9.3 | 0 (running log), 6 (assembly) | Both log, B assembles |
| No copying between groups | §9.3 | all | Joint |
| Deliverable: repository | §9.4.1 | 0–6 | Joint |
| Deliverable: 10–12 page report | §9.4.2 | 6 | Split by section |
| Deliverable: runnable system | §9.4.3 | 3, 6 | B (entrypoint), A (model loading) |
| Deliverable: demo | §9.4.4 | 6 | Joint |

---

## Open Items

- **Target device** for efficiency measurements — decide in Phase 0, record in §0.
- **Window length and hop** — A decides in Phase 1, then freezes and communicates to B.
- **Deadline and demo date** — not present in the challenge brief; update [PRD.md](PRD.md) §9.5 and pin the phase ordering to real dates once announced in class.
