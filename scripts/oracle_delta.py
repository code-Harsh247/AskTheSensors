"""Oracle-vs-real delta table (docs/TASKS.md task 3.2).

Runs every dev subject's questions through its oracle track and its real
track, and attributes each wrong answer to recognition (Member A), routing,
or reasoning (Member B). Writes <out>.json and <out>.md.

Tracks are found by name: <dir>/track_<subject>.jsonl, one per question set
in data/questions_dev/.

Usage:
    # Phase 3, with real tracks from the trained model:
    python scripts/oracle_delta.py --oracle-dir results/raw/tracks/oracle --real-dir results/raw/tracks/full

    # Dry run before a real model exists (clearly labelled as simulated).
    # Whole-burst errors are the realistic case; isolated window flips are
    # absorbed by the aggregation's per-burst vote.
    python scripts/oracle_delta.py --simulate-burst-noise 0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ats.eval.delta import render_markdown, run_delta
from ats.eval.dev import FIXTURES_DIR, QUESTIONS_DIR, REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute every lost answer to the layer that caused it.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--real-dir", type=Path, help="Directory of real-model track_<subject>.jsonl files.")
    source.add_argument("--simulate-label-noise", type=float, help="Dry run: mislabel individual oracle windows.")
    source.add_argument("--simulate-burst-noise", type=float, help="Dry run: mislabel whole oracle bursts.")
    parser.add_argument("--oracle-dir", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="Output path without extension.")
    args = parser.parse_args()

    report = run_delta(
        real_dir=args.real_dir,
        simulate_label_noise=args.simulate_label_noise,
        simulate_burst_noise=args.simulate_burst_noise,
        seed=args.seed,
        questions_dir=args.questions_dir,
        oracle_dir=args.oracle_dir,
    )
    default_name = "oracle_delta" if args.real_dir else "oracle_delta_simulated"
    out = args.out or REPO_ROOT / "results" / default_name
    out.parent.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(report)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print(f"-> {out.with_suffix('.json')}, {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
