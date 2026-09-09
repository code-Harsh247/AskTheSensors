"""CLI entry point: python -m ats.answer --questions <path> --track <path> --out <path>

Phase 1 wiring: load a window track, aggregate it into a timeline, and emit a
well-formed answer per question. Question routing and the deterministic
reasoning operators land in Phase 2 (docs/TASKS.md tasks 2B.1-2B.4), so the
answer *content* here is a single timeline-derived baseline, not a real
per-question answer. Phase 1 gates well-formedness only.
"""

from __future__ import annotations

import argparse
from typing import Any

from ats.aggregate import Timeline, build_timeline, load_track
from ats.serialize import (
    format_intervals,
    na_answer,
    read_question_set,
    write_answers,
)

PHASE1_NOTE = (
    "Phase 1 baseline: this answer reports the timeline's dominant activity "
    "rather than answering the specific question; per-question routing lands "
    "in Phase 2."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ats.answer",
        description="Answer natural-language questions about a wearable sensor recording, grounded in cited evidence.",
    )
    parser.add_argument("--recording", help="Path to the sensor recording. Not needed when --track is supplied.")
    parser.add_argument("--questions", required=True, help="Path to a question_set JSON file.")
    parser.add_argument("--track", help="Precomputed window_track JSONL (e.g. from ats.oracle) to use instead of running recognition.")
    parser.add_argument("--model", help="Model config to use for recognition, e.g. 'full', 'quant8'.")
    parser.add_argument("--out", required=True, help="Path to write answers to.")
    parser.add_argument("--format", choices=["text", "jsonl"], default="text", help="Output format.")
    return parser


def baseline_answer(question_id: str, timeline: Timeline) -> dict[str, Any]:
    """A well-formed, timeline-grounded placeholder answer."""
    activity = timeline.dominant_activity()
    if activity is None:
        return na_answer(question_id)

    intervals = [iv.as_tuple() for iv in timeline.intervals_of(activity)]
    total = timeline.total_duration(activity)
    return {
        "question_id": question_id,
        "answer": activity.replace("_", " ").title(),
        "activity_event": activity.replace("_", " ").title(),
        "evidence": {
            "timestamps": format_intervals(intervals),
            "sensor_modality": "Accelerometer, Gyroscope",
            "sensor_channels": "All",
        },
        "explanation": (
            f"{activity.replace('_', ' ').title()} accounts for the largest share of the "
            f"recording, {total:g} seconds across {len(intervals)} interval(s). {PHASE1_NOTE}"
        ),
        "tier_inferred": 1,
        "cited_intervals": [list(i) for i in intervals],
        "modality": "both",
        "channels": ["all"],
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if not args.track:
        raise NotImplementedError(
            "Running recognition from a raw recording needs Member A's model "
            "(Phase 3 integration). Supply --track for now."
        )

    timeline = build_timeline(load_track(args.track))
    question_set = read_question_set(args.questions)
    questions = question_set["questions"]

    answers = [baseline_answer(q["question_id"], timeline) for q in questions]
    queries = {q["question_id"]: q["text"] for q in questions}
    write_answers(answers, args.out, fmt=args.format, queries=queries)

    print(f"answered {len(answers)} questions -> {args.out}")


if __name__ == "__main__":
    main()
