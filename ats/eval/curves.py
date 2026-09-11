"""The data behind the accuracy figures (PRD 7.4 Figures 1 and 3; docs/TASKS.md
5B.3), kept apart from the plotting so every number is testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ats.aggregate import load_track
from ats.answer import answer_all
from ats.eval import metrics
from ats.eval.dev import dev_subjects
from ats.serialize import first_number, read_question_set

Pair = tuple[dict[str, Any], dict[str, Any]]


def answer_sets(questions_dir: Path, track_dir: Path) -> list[Pair]:
    """(question, answer) for every question set in `questions_dir`, each
    answered against its subject's track in `track_dir`."""
    pairs: list[Pair] = []
    for subject in dev_subjects(questions_dir):
        questions = read_question_set(questions_dir / f"{subject}.json")["questions"]
        answers, _ = answer_all(questions, load_track(track_dir / f"track_{subject}.jsonl"))
        pairs.extend(zip(questions, answers))
    return pairs


def iou_acceptance(pairs: Sequence[Pair], thresholds: Sequence[float]) -> tuple[int, list[float]]:
    """For every answer whose gold cites intervals: the fraction whose cited
    intervals reach each IoU threshold (matched mean IoU, PRD 7.3.3)."""
    ious = [
        metrics.matched_mean_iou(
            [tuple(i) for i in answer["cited_intervals"]],
            [tuple(i) for i in question["gold"]["cited_intervals"]],
        )
        for question, answer in pairs
        if question.get("gold", {}).get("cited_intervals")
    ]
    if not ious:
        raise ValueError("no answer's gold cites intervals")
    return len(ious), [sum(1 for iou in ious if iou >= t) / len(ious) for t in thresholds]


def tolerance_acceptance(
    pairs: Sequence[Pair], tolerances: Sequence[float], question_type: str = "duration"
) -> tuple[int, list[float]]:
    """For every numeric answer of one question type: the fraction within
    each relative tolerance of the true value. A true value of zero accepts
    only an exact zero."""
    rows = [
        (first_number(answer["answer"]), question["gold"]["numeric_value"])
        for question, answer in pairs
        if question.get("gold", {}).get("question_type") == question_type
    ]
    if not rows:
        raise ValueError(f"no {question_type} answers")
    return len(rows), [
        sum(1 for pred, gold in rows if pred is not None and metrics.within_tolerance(pred, gold, abs_tol=None, rel_tol=r))
        / len(rows)
        for r in tolerances
    ]
