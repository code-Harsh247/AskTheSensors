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
| Target device | **Laptop CPU** — AMD Ryzen 7 5800H (8C/16T), 16 GB RAM, Windows 11 64-bit, single-process CPU inference (no GPU) | Chosen for reproducibility with no cross-compilation or extra hardware; PRD §6.2 explicitly allows "a laptop processor" and says clear, consistent reporting matters more than which hardware is chosen |
| Window / hop | *(A decides in Phase 1, then frozen and communicated to B)* | Hop sets the temporal resolution of every interval boundary B produces |

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
- [ ] `pip install -e .` succeeds from a clean environment on both members' machines — **done on this machine; teammate still needs to verify on theirs**
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

  > **Status:** 52 questions generated over three **synthetic** subjects (12/16/12/12 across tiers 1–4), all four edge cases present and asserted by `tests/test_pipeline.py`. Gold is derived from the declared ground truth in `scripts/make_dev_fixture.py`, never from `ats/aggregate.py`, so using this set to test aggregation is not circular. **Questions over ≥3 real ExtraSensory subjects still to be added** once Member A publishes dev subjects — that is what the "≥3 dev subjects" requirement means and it is not yet met.

### Exit criteria

**Member A**
- [ ] `pytest tests/test_preprocess.py` passes, asserting on synthetic signals:
  - jittered input timestamps resample to **exactly** 25 Hz, with zero duplicate or backward timestamps
  - a known 1.5 Hz sinusoid survives resampling within a stated tolerance (proves we did not destroy gait-band content)
  - an injected gap produces `coverage < 1` on **exactly** the overlapping windows and no others
- [ ] `pytest tests/test_splits.py` asserts **zero subject overlap** across train/val/test
- [ ] `python -m ats.oracle --subject <id> --out track.jsonl` produces a file that validates against `window_track.schema.json`

**Member B**
- [x] `python -m ats.answer --questions data/questions_dev.json --track <track> --out ans.txt` produces **100% schema-valid output for 100% of questions**. Content correctness is *not* gated yet — well-formedness is. Gated automatically by `tests/test_pipeline.py`.
- [x] `python -m ats.eval` runs to completion and emits a metrics dict
- [x] `pytest tests/test_metrics.py` — every metric checked against a **hand-computed** fixture: a two-interval IoU worked out by hand, a macro-F1 on a toy 3-class confusion matrix, an MAPE on known values

> **Run against synthetic fixtures, not the oracle.** Member A's `ats/oracle.py` and dev subjects do not exist yet, so B's stack is currently exercised against `tests/fixtures/track_subj_synth_*.jsonl` — three synthetic subjects generated by `scripts/make_dev_fixture.py` from a declared ground truth. All three criteria must be **re-run against a real oracle track** once A's Phase 1 lands; that re-run is the first task of Phase 3.

### Artifacts crossing the boundary

- **A → B:** `ats/oracle.py` + one committed sample track + the frozen window/hop numbers
- **B → A:** `data/questions_dev.json` (**frozen at the end of this phase** — later additions go to `questions_dev_v2` so A's robustness and Pareto curves stay comparable across the project) + the importable `evaluate()`

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

### Exit criteria

**Member A**
- [ ] Window-level macro-F1 on **held-out subjects** beats two baselines computed in this same phase: (a) majority class, (b) logistic regression on per-window mean/std. The bar is *beat both*, with the margin recorded in the commit message — an absolute number invented in advance would be meaningless.
- [ ] Figure 2 generated with per-class precision/recall/F1 table alongside
- [ ] Confusions are named in writing, especially sitting vs standing-still and walking vs running (PRD §7.4.2 calls these out specifically)

**Member B** *(measured against the **oracle** track)*
- [ ] Tier-1 and tier-2 accuracy **≥ 0.95**
- [ ] Grounded accuracy at IoU 0.5 **≥ 0.90**
  > With a perfect classifier the reasoning layer has no excuse. Anything below these numbers is a bug in B's code, and testing against the oracle isolates that cleanly from recognition error.
- [ ] `pytest tests/test_validator.py` — a deliberately fabricated answer (interval absent from the timeline; duration off by 30 s) is **rejected**
- [ ] Reasoning still degrades gracefully with `--label-noise 0.1` on the oracle (no crashes, no empty answers)

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
- **3.3** Triage the delta table; each member leaves with an ordered fix list **for their own layer only**.

### Exit criteria

- [ ] `python -m ats.answer` on a dev recording with the real model produces **100% schema-valid** answers
- [ ] `scripts/oracle_delta.py` emits a per-question-type table attributing every loss to a layer
- [ ] Delta table committed to `results/` and reviewed by both members
- [ ] Both members have a written, ordered fix list

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
- **4B.4** LLM-judge rubric for explanation quality: 1–5 scale over the three PRD §7.3.5 sub-criteria (cites real signal features / features support the conclusion / conclusion plausible).
- **4B.5** **Human double-rating of a ≥20-explanation sample** to produce the inter-rater agreement number PRD §7.3.5 asks for. Both members rate independently, then compare.
- **4B.6** Instrument the interface layer using A's profiler (from 4A.2).

### Member A — Signals & Systems

- **4A.1** Work the recognition fix list from the Phase 3 delta table; focus on sedentary-class disambiguation, which is where the confusion matrix will be worst.
- **4A.2** `ats/profile.py` — the cost-measurement harness: parameter count, on-disk MB, peak RSS, per-query latency (p50/p95), CPU utilization, energy estimate. Emits `cost_report.schema.json`, naming the target device.
  > A owns the harness; **B instruments the interface layer with it**. PRD §6.1's "time taken to answer a single query" includes SLM inference, so A cannot measure it alone — and two independently written profilers would produce two incomparable numbers.

### Exit criteria

- [ ] **Ablation committed:** SLM-routed vs rule-routed accuracy on the frozen dev set. If the SLM does not beat the rules, **keep rules as the default path and say so in the report** — a measured negative result is a defensible design finding, not a failure.
- [ ] `pytest tests/test_no_ungrounded_output.py` — asserts every tier-3 and tier-4 answer carries a non-`N/A` interval that **exists in the timeline**. Zero validator rejections escape to output.
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
- **5B.3** Produce **Figure 1** (accuracy by question type) and **Figure 3** (accuracy vs strictness) from the `full` config.

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
