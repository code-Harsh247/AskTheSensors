"""ats.eval exposes `evaluate(pred, gold) -> dict` as an importable function
(docs/TASKS.md task 0.6) so the Phase 5 sweep can call it directly per
config/degradation point instead of shelling out.
"""

from __future__ import annotations

from typing import Any

from ats.eval import metrics
from ats.serialize import first_number

QUESTION_TYPES = (
    "identification",
    "verification",
    "duration",
    "count",
    "comparison",
    "grounding",
    "open_world",
)

TIER_OF_TYPE = {
    "identification": 1,
    "verification": 1,
    "duration": 2,
    "count": 2,
    "comparison": 2,
    "grounding": 3,
    "open_world": 4,
}


def _predicted_number(answer: dict[str, Any]) -> float | None:
    return first_number(answer.get("answer", ""))


def _score_one(answer: dict[str, Any] | None, gold: dict[str, Any]) -> bool:
    """Apply the correctness rule for this answer's kind (PRD §7.3). A missing
    prediction counts as wrong rather than being skipped."""
    if answer is None:
        return False

    kind = gold.get("answer_kind", "categorical")

    if kind == "numeric":
        expected = gold.get("numeric_value")
        predicted = _predicted_number(answer)
        if expected is None or predicted is None:
            return False
        return metrics.within_tolerance(predicted, expected)

    if kind == "temporal":
        pred_intervals = [tuple(i) for i in answer.get("cited_intervals", [])]
        gold_intervals = [tuple(i) for i in gold.get("cited_intervals", [])]
        return (
            metrics.matched_mean_iou(pred_intervals, gold_intervals)
            >= metrics.HEADLINE_IOU_THRESHOLD
        )

    return metrics.categorical_match(answer.get("answer", ""), gold.get("answer", ""))


def evaluate(pred: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Score predicted answers against a question set carrying gold blocks.

    `pred` is {"answers": [...]}; `gold` is a question_set dict. Questions
    without a gold block are skipped and counted in `skipped_no_gold`.
    Returns a JSON-serialisable metrics dict.
    """
    answers_by_id = {a["question_id"]: a for a in pred.get("answers", [])}

    graded: list[tuple[dict[str, Any], dict[str, Any] | None, bool]] = []
    skipped = 0
    for question in gold.get("questions", []):
        gold_block = question.get("gold")
        if not gold_block:
            skipped += 1
            continue
        answer = answers_by_id.get(question["question_id"])
        graded.append((gold_block, answer, _score_one(answer, gold_block)))

    if not graded:
        return {
            "n_graded": 0,
            "skipped_no_gold": skipped,
            "note": "no gold blocks present; nothing to score",
        }

    by_type: dict[str, dict[str, Any]] = {}
    for question_type in QUESTION_TYPES:
        rows = [g for g in graded if g[0].get("question_type") == question_type]
        if not rows:
            continue
        correct = sum(1 for _, _, ok in rows if ok)
        entry: dict[str, Any] = {"n": len(rows), "accuracy": correct / len(rows)}

        usable = [
            (_predicted_number(a), g.get("numeric_value"))
            for g, a, _ in rows
            if g.get("answer_kind") == "numeric" and a is not None
        ]
        usable = [(p, e) for p, e in usable if p is not None and e is not None]
        if usable:
            entry["mae"] = metrics.mae([p for p, _ in usable], [e for _, e in usable])

        if question_type == "verification":
            preds = [a.get("answer", "") if a else "" for _, a, _ in rows]
            golds = [g.get("answer", "") for g, _, _ in rows]
            entry["binary"] = metrics.binary_verification(preds, golds)

        by_type[question_type] = entry

    by_tier: dict[str, dict[str, Any]] = {}
    for tier in (1, 2, 3, 4):
        oks = [ok for g, _, ok in graded if TIER_OF_TYPE.get(g.get("question_type", "")) == tier]
        if oks:
            by_tier[str(tier)] = {"n": len(oks), "accuracy": sum(oks) / len(oks)}

    grounded_rows = [(g, a, ok) for g, a, ok in graded if g.get("cited_intervals")]
    grounded_accuracy = None
    if grounded_rows:
        grounded_hits = sum(
            1
            for g, a, ok in grounded_rows
            if a is not None and metrics.is_grounded_and_correct(a, g, ok)
        )
        grounded_accuracy = grounded_hits / len(grounded_rows)

    overall_correct = sum(1 for _, _, ok in graded if ok)
    return {
        "n_graded": len(graded),
        "skipped_no_gold": skipped,
        "n_missing_predictions": sum(1 for _, a, _ in graded if a is None),
        "by_question_type": by_type,
        "by_tier": by_tier,
        "overall_micro_accuracy": overall_correct / len(graded),
        "overall_macro_accuracy": metrics.macro_average(
            [e["accuracy"] for e in by_type.values()]
        ),
        "grounded_accuracy": grounded_accuracy,
        "n_grounded_rows": len(grounded_rows),
        "iou_threshold": metrics.HEADLINE_IOU_THRESHOLD,
        "relative_tolerance": metrics.RELATIVE_DURATION_TOL,
        "absolute_tolerance_s": metrics.DURATION_ABS_TOL_S,
    }
