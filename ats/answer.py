"""CLI entry point: python -m ats.answer --recording <path> --questions <path> --out <path>

Stub for Phase 0 (contract freeze). Real routing/aggregation/reasoning
lands in Phase 1-4; see docs/TASKS.md.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ats.answer",
        description="Answer natural-language questions about a wearable sensor recording, grounded in cited evidence.",
    )
    parser.add_argument("--recording", required=True, help="Path to the sensor recording.")
    parser.add_argument("--questions", required=True, help="Path to a question_set JSON file.")
    parser.add_argument("--track", help="Optional precomputed window_track (e.g. from ats.oracle) to use instead of running the recognition model.")
    parser.add_argument("--model", help="Model config to use for recognition, e.g. 'full', 'quant8'.")
    parser.add_argument("--out", required=True, help="Path to write answers to.")
    parser.add_argument("--format", choices=["text", "jsonl"], default="text", help="Output format.")
    return parser


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    raise NotImplementedError(
        "ats.answer is a Phase 0 stub. Aggregation, routing, and reasoning land in Phase 1-4 (docs/TASKS.md)."
    )


if __name__ == "__main__":
    main()
