"""CLI entry point, the system graders run:

    python -m ats.answer --recording <data_dir> --subject <id> --questions <path> --out <path>
    python -m ats.answer --track <window_track.jsonl> --questions <path> --out <path>

--recording runs Member A's recognition pipeline (ats.recognize) on a raw
ExtraSensory-layout recording to build the window track; --track skips it with
a track built earlier (e.g. by ats.oracle). --subject is needed because an
ExtraSensory data directory can hold many subjects.

Route each question to a typed operator, compute the answer deterministically
from the activity timeline, attach the evidence behind it, and pass it
through the grounding validator before it can be written. An answer the
validator rejects is never emitted: it is replaced by an explicit abstention
that is itself validated, and the rejection is reported rather than dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from ats.aggregate import Timeline, build_timeline, load_track
from ats.evidence import CHANNELS, CHANNELS_TEXT, MODALITY, MODALITY_TEXT
from ats.operators import Finding, abstain, execute
from ats.routing import OperatorCall, route
from ats.serialize import format_intervals, read_question_set, write_answers
from ats.validator import grounding_problems, validate_grounding

Router = Callable[[str], OperatorCall]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ats.answer",
        description="Answer natural-language questions about a wearable sensor recording, grounded in cited evidence.",
    )
    parser.add_argument(
        "--recording",
        help="ExtraSensory-layout data directory holding the raw recording (the directory scripts/fetch_data.py writes, with _meta/ inside).",
    )
    parser.add_argument("--subject", help="Subject ID of the recording inside --recording.")
    parser.add_argument("--questions", required=True, help="Path to a question_set JSON file.")
    parser.add_argument("--track", help="Precomputed window_track JSONL (e.g. from ats.oracle) to use instead of running recognition.")
    parser.add_argument("--model", default="models/full/activity_cnn.pt", help="Trained recognition model weights (.pt) for --recording.")
    parser.add_argument("--model-id", default="full", help="model_id stamped on the window track built from --recording.")
    parser.add_argument("--save-track", help="Also write the window track built from --recording to this JSONL path.")
    parser.add_argument("--out", required=True, help="Path to write answers to.")
    parser.add_argument("--format", choices=["text", "jsonl"], default="text", help="Output format.")
    parser.add_argument(
        "--router",
        choices=["rules", "slm"],
        default="rules",
        help="Question parser: the rule router, or the SLM with the rule router as fallback.",
    )
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
    question: dict[str, Any],
    timeline: Timeline,
    windows: Sequence[dict[str, Any]],
    router: Router = route,
) -> tuple[dict[str, Any], list[str]]:
    """Returns the answer to emit and any grounding problems found in the
    answer that was originally computed."""
    call = router(question["text"])
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
    questions: Sequence[dict[str, Any]],
    windows: Sequence[dict[str, Any]],
    router: Router = route,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timeline = build_timeline(windows)
    answers: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for question in questions:
        answer, problems = answer_question(question, timeline, windows, router)
        answers.append(answer)
        if problems:
            rejections.append({"question_id": question["question_id"], "problems": problems})
    return answers, rejections


def _recognise(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build the window track from a raw recording with the trained model.
    Imported here so the --track path never needs torch."""
    from ats.recognize import build_track, load_model

    windows = build_track(args.subject, args.recording, load_model(args.model), model_id=args.model_id)
    if args.save_track:
        path = Path(args.save_track)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for window in windows:
                f.write(json.dumps(window, sort_keys=True) + "\n")
    return windows


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if bool(args.track) == bool(args.recording):
        parser.error("supply exactly one of --recording or --track")
    if args.recording and not args.subject:
        parser.error("--recording needs --subject, the recording's subject ID")

    windows = load_track(args.track) if args.track else _recognise(args)
    questions = read_question_set(args.questions)["questions"]
    router: Router = route
    if args.router == "slm":
        from ats.slm import SLMRouter

        router = SLMRouter()
    answers, rejections = answer_all(questions, windows, router)
    write_answers(answers, args.out, fmt=args.format, queries={q["question_id"]: q["text"] for q in questions})

    for rejection in rejections:
        print(f"withheld {rejection['question_id']}: {'; '.join(rejection['problems'])}", file=sys.stderr)
    print(f"answered {len(answers)} questions ({len(rejections)} withheld by the grounding validator) -> {args.out}")
    if args.router == "slm":
        print(f"SLM parsed {router.n_parsed} questions; the rule router took over for {router.n_fallback}")


if __name__ == "__main__":
    main()
