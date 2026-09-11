"""Phase 5 sweep (docs/TASKS.md task 5A.3): every compressed config against
the frozen question set (results/pareto.csv), and one config across several
degradation levels (results/robustness.csv). Calls `ats.eval.evaluate()`
directly as a function -- no subprocess, no CLI round trip -- so hundreds of
sweep points stay fast and don't depend on writing/reading files in between.

The question set and its scoring rules are frozen for the duration of this
sweep (docs/TASKS.md task 5B.1, Member B): every point must be comparable.

Cost columns come from each config's standalone cost report
(`python -m ats.profile --config <config>`, results/cost_report_<config>.json),
not from timing inside this process: with the whole dataset loaded, a short
in-sweep timing run inflated latency (quant8 read 1.37 ms here against 0.35 ms
standalone) and its peak memory was the sweep process's, not the model's.

Requires `python scripts/fetch_data.py` (raw archives), the compressed
configs already built (`python -m ats.compress --config quant8|pruned30|pruned60`),
and a cost report for each config.

Usage: python scripts/sweep.py [--data-dir data/raw] [--questions-dir data/questions_dev_v2]
       python scripts/sweep.py --refresh-costs   # re-read cost columns only; no raw data needed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ats.answer import answer_all  # noqa: E402
from ats.degrade import degrade_global_samples  # noqa: E402
from ats.eval import evaluate  # noqa: E402
from ats.ingest import load_subject  # noqa: E402
from ats.recognize import build_track_from_globalized, load_model  # noqa: E402
from ats.resample import GlobalSamples, globalize_subject  # noqa: E402
from ats.serialize import read_question_set  # noqa: E402
from scripts.make_real_dev_questions import UUIDS  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_QUESTIONS_DIR = REPO_ROOT / "data" / "questions_dev_v2"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
PARETO_OUT = RESULTS_DIR / "pareto.csv"
ROBUSTNESS_OUT = RESULTS_DIR / "robustness.csv"

CONFIGS = ("full", "quant8", "pruned30", "pruned60")
# The cost columns of results/pareto.csv, as named in the cost reports.
COST_KEYS = ("params", "disk_mb", "peak_rss_mb", "latency_p50_ms", "latency_p95_ms", "cpu_pct")

# One robustness point per (kind, level); "none" is the shared baseline row
# (the same clean windows the "full" pareto point already used). PRD Sec
# 7.4.5 names three axes -- ats/degrade.py supports all three -- but only
# one needs >=4 levels to meet the exit criterion; dropout is run by
# default to keep the sweep's wall-clock time down (each level re-runs
# full recognition + reasoning over both real subjects). Pass --all-axes to
# additionally sweep noise and decimation.
ROBUSTNESS_AXES: dict[str, list[float]] = {
    "dropout": [0.1, 0.3, 0.5, 0.7],  # fraction of samples dropped
}
EXTRA_ROBUSTNESS_AXES: dict[str, list[float]] = {
    "noise": [20.0, 10.0, 5.0, 0.0],  # target SNR, dB (lower = noisier)
    "decimate": [2, 4, 8, 16],  # keep-every-Nth-sample factor
}


def costs_from_report(config: str, results_dir: Path = RESULTS_DIR) -> dict:
    """A config's cost columns, from its standalone cost report."""
    path = results_dir / f"cost_report_{config}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run `python -m ats.profile --config {config}` first")
    report = json.loads(path.read_text(encoding="utf-8"))
    return {key: report[key] for key in COST_KEYS}


def refresh_costs(pareto_path: Path = PARETO_OUT, results_dir: Path = RESULTS_DIR) -> list[dict]:
    """The existing pareto rows with their measured accuracy kept and every
    cost column re-read from the cost reports. Needs no raw data."""
    with pareto_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {"config_id": row["config_id"], "accuracy": float(row["accuracy"]), **costs_from_report(row["config_id"], results_dir)}
        for row in rows
    ]


def _load_globalized(data_dir: Path) -> dict[str, GlobalSamples]:
    globalized = {}
    for alias, uuid in UUIDS.items():
        subject = load_subject(data_dir, uuid)
        globalized[alias] = globalize_subject(subject)
    return globalized


def _accuracy_for(globalized: dict[str, GlobalSamples], model, model_id: str, questions_dir: Path) -> float:
    """Builds a track per subject from `globalized` (already
    loaded/globalized/possibly-degraded) with `model`, answers each
    subject's own frozen question set, and scores everything together --
    the same combine-then-evaluate convention as ats.eval.curves.answer_sets
    and scripts/make_fig1.py, so this number means the same thing they do."""
    all_answers, all_questions = [], []
    for alias, g in globalized.items():
        questions = read_question_set(questions_dir / f"{alias}.json")["questions"]
        windows = build_track_from_globalized(g, model, model_id=model_id, subject_id=alias)
        answers, _ = answer_all(questions, windows)
        all_answers.extend(answers)
        all_questions.extend(questions)
    report = evaluate({"answers": all_answers}, {"questions": all_questions})
    return report["overall_macro_accuracy"]


def run_pareto(globalized: dict[str, GlobalSamples], questions_dir: Path) -> list[dict]:
    rows = []
    for config in CONFIGS:
        model_path = MODELS_DIR / config / "activity_cnn.pt"
        if not model_path.is_file():
            print(f"[pareto] skipping {config}: {model_path} not built yet")
            continue
        model = load_model(model_path)
        accuracy = _accuracy_for(globalized, model, config, questions_dir)
        cost = costs_from_report(config)
        rows.append({"config_id": config, "accuracy": accuracy, **cost})
        print(f"[pareto] {config}: accuracy={accuracy:.4f} disk_mb={cost['disk_mb']:.4f} latency_p50_ms={cost['latency_p50_ms']:.3f}")
    return rows


def run_robustness(globalized: dict[str, GlobalSamples], questions_dir: Path, all_axes: bool = False) -> list[dict]:
    model_path = MODELS_DIR / "full" / "activity_cnn.pt"
    model = load_model(model_path)

    rows = [{"config_id": "full", "axis": "none", "level": 0.0, "accuracy": _accuracy_for(globalized, model, "full", questions_dir)}]
    print(f"[robustness] none (baseline): accuracy={rows[0]['accuracy']:.4f}")

    axes = {**ROBUSTNESS_AXES, **EXTRA_ROBUSTNESS_AXES} if all_axes else ROBUSTNESS_AXES
    for axis, levels in axes.items():
        for level in levels:
            degraded = {alias: degrade_global_samples(g, axis, level, seed=0) for alias, g in globalized.items()}
            accuracy = _accuracy_for(degraded, model, "full", questions_dir)
            rows.append({"config_id": "full", "axis": axis, "level": level, "accuracy": accuracy})
            print(f"[robustness] {axis}={level}: accuracy={accuracy:.4f}")
    return rows


def _write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {out_path}")
    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    try:
        shown = out_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_path
    print(f"wrote {len(rows)} rows -> {shown}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--questions-dir", default=str(DEFAULT_QUESTIONS_DIR))
    parser.add_argument("--all-axes", action="store_true", help="Also sweep noise and decimation, not just dropout.")
    parser.add_argument("--skip-pareto", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument(
        "--refresh-costs",
        action="store_true",
        help="Only re-read results/pareto.csv's cost columns from the cost reports, keeping its accuracies.",
    )
    args = parser.parse_args(argv)

    if args.refresh_costs:
        _write_csv(refresh_costs(), PARETO_OUT)
        return

    globalized = _load_globalized(Path(args.data_dir))

    if not args.skip_pareto:
        pareto_rows = run_pareto(globalized, Path(args.questions_dir))
        _write_csv(pareto_rows, PARETO_OUT)

    if not args.skip_robustness:
        robustness_rows = run_robustness(globalized, Path(args.questions_dir), all_axes=args.all_axes)
        _write_csv(robustness_rows, ROBUSTNESS_OUT)


if __name__ == "__main__":
    main()
