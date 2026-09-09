"""The PRD §7.3 correctness rules, one function per rule.

Owned by Member B (docs/TASKS.md task 1B.3). Metrics raise on empty input
rather than returning a placeholder, so an empty question group surfaces as
a bug instead of a silent zero.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

Interval = tuple[float, float]

# Pre-registered in docs/TASKS.md §0 before any results existed, so the
# choice cannot be tuned to flatter our numbers.
HEADLINE_IOU_THRESHOLD = 0.5
RELATIVE_DURATION_TOL = 0.10

# The absolute half of the frozen tolerance is max(2 x window_hop, 10% rel).
# The hop is Member A's Phase 1 decision, so it stays unset until published;
# until then the relative tolerance alone applies.
DURATION_ABS_TOL_S: float | None = None


def _require(values: Sequence, name: str = "input") -> None:
    if len(values) == 0:
        raise ValueError(f"{name} is empty; a metric over zero items is undefined")


def normalize_categorical(value: str) -> str:
    """Case- and whitespace-insensitive comparison key, trailing period dropped."""
    return re.sub(r"\s+", " ", value.strip().lower()).rstrip(".")


def categorical_match(pred: str, gold: str) -> bool:
    return normalize_categorical(pred) == normalize_categorical(gold)


# --- Categorical answers (PRD §7.3.1) ---------------------------------------


def accuracy(preds: Sequence[str], golds: Sequence[str]) -> float:
    _require(preds, "preds")
    if len(preds) != len(golds):
        raise ValueError("preds and golds must be the same length")
    hits = sum(1 for p, g in zip(preds, golds) if categorical_match(p, g))
    return hits / len(preds)


def confusion_matrix(
    preds: Sequence[str], golds: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    """Rows are true labels, columns predicted, both in `labels` order."""
    _require(preds, "preds")
    index = {normalize_categorical(l): i for i, l in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for pred, gold in zip(preds, golds):
        g = index.get(normalize_categorical(gold))
        p = index.get(normalize_categorical(pred))
        if g is None:
            raise ValueError(f"gold label {gold!r} is not in labels")
        if p is not None:
            matrix[g][p] += 1
    return matrix


def per_class_prf(
    preds: Sequence[str], golds: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, float]]:
    _require(preds, "preds")
    result: dict[str, dict[str, float]] = {}
    for label in labels:
        key = normalize_categorical(label)
        tp = sum(
            1
            for p, g in zip(preds, golds)
            if normalize_categorical(p) == key and normalize_categorical(g) == key
        )
        fp = sum(
            1
            for p, g in zip(preds, golds)
            if normalize_categorical(p) == key and normalize_categorical(g) != key
        )
        fn = sum(
            1
            for p, g in zip(preds, golds)
            if normalize_categorical(p) != key and normalize_categorical(g) == key
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    return result


def macro_f1(
    preds: Sequence[str], golds: Sequence[str], labels: Sequence[str]
) -> float:
    scores = per_class_prf(preds, golds, labels)
    return sum(s["f1"] for s in scores.values()) / len(labels)


def balanced_accuracy(
    preds: Sequence[str], golds: Sequence[str], labels: Sequence[str]
) -> float:
    """Mean per-class recall, restricted to classes that actually occur."""
    scores = per_class_prf(preds, golds, labels)
    present = [s for s in scores.values() if s["support"] > 0]
    if not present:
        raise ValueError("no gold label occurs in the evaluated set")
    return sum(s["recall"] for s in present) / len(present)


def binary_verification(
    preds: Sequence[str], golds: Sequence[str], positive: str = "yes"
) -> dict[str, float]:
    """Precision/recall/F1 on the positive class plus specificity, because
    plain accuracy hides a bias toward always answering no (PRD §7.3.1)."""
    _require(preds, "preds")
    pos = normalize_categorical(positive)
    tp = fp = fn = tn = 0
    for pred, gold in zip(preds, golds):
        p_is = normalize_categorical(pred) == pos
        g_is = normalize_categorical(gold) == pos
        if p_is and g_is:
            tp += 1
        elif p_is and not g_is:
            fp += 1
        elif not p_is and g_is:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": (tp + tn) / len(preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# --- Numeric answers (PRD §7.3.2) -------------------------------------------


def within_tolerance(
    pred: float,
    gold: float,
    abs_tol: float | None = DURATION_ABS_TOL_S,
    rel_tol: float = RELATIVE_DURATION_TOL,
) -> bool:
    error = abs(pred - gold)
    allowed = abs(gold) * rel_tol
    if abs_tol is not None:
        allowed = max(allowed, abs_tol)
    return error <= allowed


def tolerance_accuracy(
    preds: Sequence[float],
    golds: Sequence[float],
    abs_tol: float | None = DURATION_ABS_TOL_S,
    rel_tol: float = RELATIVE_DURATION_TOL,
) -> float:
    _require(preds, "preds")
    hits = sum(
        1 for p, g in zip(preds, golds) if within_tolerance(p, g, abs_tol, rel_tol)
    )
    return hits / len(preds)


def mae(preds: Sequence[float], golds: Sequence[float]) -> float:
    _require(preds, "preds")
    return sum(abs(p - g) for p, g in zip(preds, golds)) / len(preds)


def mape(preds: Sequence[float], golds: Sequence[float]) -> float:
    """Mean absolute percentage error as a fraction. Zero-valued golds are
    skipped, since the percentage is undefined there."""
    pairs = [(p, g) for p, g in zip(preds, golds) if g != 0]
    _require(pairs, "non-zero-gold pairs")
    return sum(abs(p - g) / abs(g) for p, g in pairs) / len(pairs)


# --- Temporal answers and cited intervals (PRD §7.3.3) ----------------------


def _length(interval: Interval) -> float:
    return max(0.0, interval[1] - interval[0])


def _overlap(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_iou(a: Interval, b: Interval) -> float:
    intersection = _overlap(a, b)
    union = _length(a) + _length(b) - intersection
    return intersection / union if union > 0 else 0.0


def temporal_prf(
    pred_intervals: Sequence[Interval], gold_intervals: Sequence[Interval]
) -> dict[str, float]:
    """Overlap-based precision/recall/F1 for multi-interval answers: total
    overlap over predicted length, and over true length."""
    pred_len = sum(_length(i) for i in pred_intervals)
    gold_len = sum(_length(i) for i in gold_intervals)
    overlap = sum(_overlap(p, g) for p in pred_intervals for g in gold_intervals)
    precision = overlap / pred_len if pred_len > 0 else 0.0
    recall = overlap / gold_len if gold_len > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def matched_mean_iou(
    pred_intervals: Sequence[Interval], gold_intervals: Sequence[Interval]
) -> float:
    """Greedily match predicted to gold intervals by IoU and average. Unmatched
    gold intervals score 0, so under-prediction is penalised."""
    if not gold_intervals:
        return 1.0 if not pred_intervals else 0.0
    remaining = list(pred_intervals)
    scores: list[float] = []
    for gold in gold_intervals:
        if not remaining:
            scores.append(0.0)
            continue
        best = max(remaining, key=lambda p: interval_iou(p, gold))
        score = interval_iou(best, gold)
        scores.append(score)
        if score > 0:
            remaining.remove(best)
    return sum(scores) / len(scores)


def grounding_accuracy(
    pred_sets: Sequence[Sequence[Interval]],
    gold_sets: Sequence[Sequence[Interval]],
    threshold: float = HEADLINE_IOU_THRESHOLD,
) -> float:
    """Fraction of answers whose cited intervals reach the IoU threshold. This
    is the curve traced by the accuracy-versus-strictness figure."""
    _require(pred_sets, "pred_sets")
    hits = sum(
        1
        for pred, gold in zip(pred_sets, gold_sets)
        if matched_mean_iou(pred, gold) >= threshold
    )
    return hits / len(pred_sets)


# --- Evidence grounding as a whole (PRD §7.3.4) -----------------------------


def channels_match(pred: Sequence[str], gold: Sequence[str]) -> bool:
    """'all' matches any concrete channel list, since citing every channel is
    a superset of citing some of them."""
    p = {normalize_categorical(c) for c in pred}
    g = {normalize_categorical(c) for c in gold}
    if "all" in p or "all" in g:
        return True
    return p == g


def is_grounded_and_correct(
    answer: dict,
    gold: dict,
    answer_correct: bool,
    threshold: float = HEADLINE_IOU_THRESHOLD,
) -> bool:
    """An answer counts as grounded only when it is correct, its cited interval
    reaches the IoU threshold, and its modality and channels match."""
    if not answer_correct:
        return False
    pred_intervals = [tuple(i) for i in answer.get("cited_intervals", [])]
    gold_intervals = [tuple(i) for i in gold.get("cited_intervals", [])]
    if matched_mean_iou(pred_intervals, gold_intervals) < threshold:
        return False
    if normalize_categorical(answer.get("modality", "")) != normalize_categorical(
        gold.get("modality", "")
    ):
        return False
    return channels_match(answer.get("channels", []), gold.get("channels", []))


def grounding_precision(
    pred_sets: Sequence[Sequence[Interval]],
    gold_sets: Sequence[Sequence[Interval]],
) -> float:
    """Lighter check needing no interval labels: fraction of answers whose
    cited interval overlaps the true activity at all (PRD §7.3.4)."""
    _require(pred_sets, "pred_sets")
    hits = 0
    for pred, gold in zip(pred_sets, gold_sets):
        if any(_overlap(p, g) > 0 for p in pred for g in gold):
            hits += 1
    return hits / len(pred_sets)


def macro_average(values: Sequence[float]) -> float:
    """Macro-average across question types, so abundant easy sedentary
    examples do not dominate the headline score (PRD §7.3)."""
    usable = [v for v in values if not math.isnan(v)]
    _require(usable, "values")
    return sum(usable) / len(usable)
