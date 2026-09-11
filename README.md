# Ask the Sensors

Answers plain-language questions about a person's activity from a phone's accelerometer and gyroscope, and returns every answer with the stretch of signal it rests on. CS60055 Ubiquitous Computing, IIT Kharagpur, Hackathon Challenge 1.

A compact 1D-CNN labels 4-second windows of 25 Hz signal. Everything after that is deterministic Python: the windows become an activity timeline, a router maps each question to an operator, the operator computes the answer and its evidence, and a validator refuses any answer whose cited intervals or numbers do not check out. No language model produces a number, an interval or a verdict. Design and results: [docs/report/report.pdf](docs/report/report.pdf).

## Setup

Python 3.10 or newer.

```bash
pip install -e ".[dev]"
```

Optional: `pip install -e ".[slm]"` adds the small-language-model question parser (`--router slm`), which downloads the Qwen2.5-0.5B weights (about 1 GB) on first use. The default rule router needs none of it.

The trained recognition weights are committed at `models/full/activity_cnn.pt`, so answering questions needs no training.

## Data and training

> **Member A to complete and check this section.** What follows is taken from the scripts' own documentation.

The raw ExtraSensory data is never committed. Fetch it (about 15 GB: accelerometer, gyroscope and labels for all 60 users) into `data/raw`:

```bash
python scripts/fetch_data.py
```

`python scripts/fetch_data.py --list-subjects` prints the 60 subject IDs. Building the feature dataset and retraining the model: `scripts/build_feature_dataset.py`, `scripts/make_splits.py` and `scripts/train_cnn.py` (training was run on Kaggle); Figure 2 comes from `scripts/make_fig2.py`.

## Answering questions

### The question file

A JSON file with a list of questions. Each needs an ID and the text; the `gold` block is optional and only used for evaluation.

```json
{
  "questions": [
    {"question_id": "q1", "text": "How long was the user walking?"},
    {"question_id": "q2", "text": "Did the user lie down for a prolonged period?"}
  ]
}
```

### From a raw recording (evaluation time)

Point `--recording` at a folder in the ExtraSensory layout (the `data/raw` folder `fetch_data.py` writes, with `_meta/` inside) and name the subject. Labels are not needed.

```bash
python -m ats.answer --recording data/raw --subject <SUBJECT_ID> --questions questions.json --out answers.txt
```

This runs recognition with the committed model and answers every question. Add `--save-track track.jsonl` to keep the per-window predictions.

### From a precomputed track

Skip recognition by answering from a window track, for example the committed real-model tracks:

```bash
python -m ats.answer --track tests/fixtures/real_model_tracks/track_subj_real_a.jsonl --questions data/questions_dev_v2/subj_real_a.json --out answers.txt
```

### The output

Every answer uses the brief's fields, in this order. Times are seconds from the start of the recording.

```
Query: "Was the user resting for a long time?"
Answer:              Likely yes
Activity/Event:      Sustained stillness
Evidence:
    Timestamp(s):        5758 to 6548, 7579 to 8249, ...
    Sensor Modality:     Accelerometer, Gyroscope
    Sensor Channel(s):   All
Explanation:         The longest stretch of mostly still signal runs 216560-237464 s (20904 s), ...
```

When the system cannot support an answer it returns `N/A` with the reason in the explanation, rather than guessing. `--format jsonl` writes one JSON answer per line, with the cited intervals as numbers, which is what the evaluator reads.

## Evaluating

Score answers against a question file that has gold blocks:

```bash
python -m ats.answer --track tests/fixtures/real_model_tracks/track_subj_real_a.jsonl --questions data/questions_dev_v2/subj_real_a.json --out answers.jsonl --format jsonl
python -m ats.eval --pred answers.jsonl --gold data/questions_dev_v2/subj_real_a.json --out eval/
```

`eval/metrics.json` has accuracy by question type and tier, the macro-averaged overall score, grounded accuracy at IoU 0.5, verification precision/recall/F1/specificity, and duration and count errors. The same function is importable as `ats.eval.evaluate(pred, gold)`.

## Reproducing the reported results

Everything below runs from a clean checkout without the raw data, using the committed tracks.

| Result | Command | Output |
|---|---|---|
| Reasoning on ground-truth labels, 53 synthetic questions | `python scripts/run_dev_eval.py` | `results/phase2_dev_eval.json` |
| Same, with 10% of bursts mislabelled | `python scripts/run_dev_eval.py --burst-noise 0.1 --out results/phase2_dev_eval_burst10.json` | `results/phase2_dev_eval_burst10.json` |
| Oracle vs trained model on two real subjects, losses by layer | `python scripts/oracle_delta.py --oracle-dir tests/fixtures --real-dir tests/fixtures/real_model_tracks --questions-dir data/questions_dev_v2` | `results/oracle_delta.md` |
| Figure 1, accuracy by question type | `python scripts/make_fig1.py` | `results/fig1_*` |
| Figure 3, accuracy vs strictness | `python scripts/make_fig3.py` | `results/fig3_*` |
| Stillness thresholds and their check | `python scripts/calibrate_stillness.py` | `results/stillness_calibration.json` |
| Open-world probe, 16 questions | `python scripts/probe_open_world.py` | printed |
| Explanation scores and judge agreement | `python scripts/judge_explanations.py` | `results/rubric_summary.json` |
| Rules vs SLM router ablation | `python scripts/run_router_ablation.py` (needs `.[slm]`) | `results/phase4_router_ablation.json` |
| Recognition model cost | `python -m ats.profile --config full` | `results/cost_report_full.json` |
| Answering-layer latency | `python scripts/profile_pipeline.py` | `results/pipeline_latency_rules.json` |

The explanation scores are rebuilt from the committed judgments in `results/rubric_judgments.jsonl` with no API calls. Re-judging from scratch needs an OpenRouter key in `OPENROUTER_API_KEY` and `--yes`. Latency and cost numbers are only comparable on the target device named in the report.

## Tests

```bash
pytest
```

One test loads the pinned Qwen weights, and is skipped unless they have already been downloaded (for example by a `--router slm` run).

## Repository layout

| Path | What |
|---|---|
| `ats/` | the package: ingestion, resampling, windowing and the model (signal path); aggregation, routing, operators, evidence, validator and the SLM parser (reasoning path); `ats/eval/` the metrics |
| `schemas/` | the JSON schemas for window tracks, answers, question sets and cost reports |
| `scripts/` | data, training, evaluation and figure scripts |
| `data/questions_dev*/` | our development question sets (the raw dataset itself is not committed) |
| `tests/` | tests, and committed fixture tracks under `tests/fixtures/` |
| `results/` | committed result files and figures |
| `docs/` | the report, requirements (`PRD.md`), plan (`TASKS.md`) and citations (`CITATIONS.md`) |
