"""CLI entry point: python -m ats.eval --pred <path> --gold <path> --out <dir>

Stub for Phase 0 (contract freeze). See docs/TASKS.md.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ats.eval",
        description="Score predicted answers against gold and write the PRD sec 7.4 figures/metrics.",
    )
    parser.add_argument("--pred", required=True, help="Path to predicted answers (jsonl).")
    parser.add_argument("--gold", required=True, help="Path to a question_set JSON file with gold blocks.")
    parser.add_argument("--out", required=True, help="Directory to write metrics/figures to.")
    return parser


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    raise NotImplementedError(
        "ats.eval CLI is a Phase 0 stub. The metric library lands in Phase 1 (docs/TASKS.md task 1B.3)."
    )


if __name__ == "__main__":
    main()
