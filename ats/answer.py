"""CLI entry point: python -m ats.answer --questions <path> --track <path> --out <path>

Route each question to a typed operator, compute the answer deterministically
from the activity timeline, attach the evidence behind it, and pass it
through the grounding validator before it can be written. An answer the
validator rejects is never emitted: it is replaced by an explicit abstention
that is itself validated, and the rejection is reported rather than dropped.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from ats.aggregate import Timeline, build_timeline, load_track
from ats.evidence import CHANNELS, CHANNELS_TEXT, MODALITY, MODALITY_TEXT
from ats.operators import Finding, abstain, execute
from ats.routing import OperatorCall, route
from ats.serialize import format_intervals, read_question_set, write_answers
from ats.validator import grounding_problems, validate_grounding


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


def to_answer(question_id: str, call: OperatorCall, finding: Finding) -> dict[str, Any]:
    cited = [list(span) for span in finding.cited]
    has_evidence = bool(cited)
    return {
        "question_id": question_id,
        "answer": finding.answer,
        "activity_event": finding.activity_event,
        "evidence": {
            "timestamps": format_intervals(finding.cited),
            "sensor_modality": MODALITY_TEXT if has_evidence else "N/A",
            "sensor_channels": CHANNELS_TEXT if has_evidence else "N/A",
        },
        "explanation": finding.explanation,
        "tier_inferred": call.tier,
        "cited_intervals": cited,
        "modality": MODALITY if has_evidence else "N/A",
        "channels": list(CHANNELS) if has_evidence else ["N/A"],
    }


def answer_question(
    question: dict[str, Any], timeline: Timeline, windows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Returns the answer to emit and any grounding problems found in the
    answer that was originally computed."""
    call = route(question["text"])
    answer = to_answer(question["question_id"], call, execute(call, timeline, windows))
    problems = grounding_problems(answer, call, timeline)
    if problems:
        answer = to_answer(
            question["question_id"],
            call,
            abstain(f"Withheld: the computed answer failed grounding validation ({'; '.join(problems)})."),
        )
        validate_grounding(answer, call, timeline)
    return answer, problems


def answer_all(
    questions: Sequence[dict[str, Any]], windows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timeline = build_timeline(windows)
    answers: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for question in questions:
        answer, problems = answer_question(question, timeline, windows)
        answers.append(answer)
        if problems:
            rejections.append({"question_id": question["question_id"], "problems": problems})
    return answers, rejections


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if not args.track:
        raise NotImplementedError(
            "Running recognition from a raw recording needs Member A's model "
            "(Phase 3 integration). Supply --track for now."
        )

    windows = load_track(args.track)
    questions = read_question_set(args.questions)["questions"]
    answers, rejections = answer_all(questions, windows)
    write_answers(answers, args.out, fmt=args.format, queries={q["question_id"]: q["text"] for q in questions})

    for rejection in rejections:
        print(f"withheld {rejection['question_id']}: {'; '.join(rejection['problems'])}", file=sys.stderr)
    print(f"answered {len(answers)} questions ({len(rejections)} withheld by the grounding validator) -> {args.out}")


if __name__ == "__main__":
    main()
