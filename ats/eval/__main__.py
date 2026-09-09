"""CLI entry point: python -m ats.eval --pred <path> --gold <path> --out <dir>"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ats.eval import evaluate
from ats.serialize import read_answers_jsonl, read_question_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ats.eval",
        description="Score predicted answers against gold and write the PRD sec 7.4 metrics.",
    )
    parser.add_argument("--pred", required=True, help="Path to predicted answers (jsonl).")
    parser.add_argument("--gold", required=True, help="Path to a question_set JSON file with gold blocks.")
    parser.add_argument("--out", required=True, help="Directory to write metrics to.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    answers = read_answers_jsonl(args.pred)
    question_set = read_question_set(args.gold)
    report = evaluate({"answers": answers}, question_set)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"graded {report['n_graded']} questions -> {metrics_path}")


if __name__ == "__main__":
    main()
