"""Figure data (PRD 7.4 Figures 1, 3 and 4) checked against hand-worked
fixtures."""

import pytest

from ats.eval import score_answer
from ats.eval.curves import iou_acceptance, tolerance_acceptance
from ats.eval.pareto import pareto_frontier


def test_pareto_frontier_keeps_only_non_dominated_points():
    points = [
        {"id": "a", "mb": 1.0, "acc": 0.50},
        {"id": "b", "mb": 2.0, "acc": 0.70},
        {"id": "c", "mb": 3.0, "acc": 0.60},  # dominated by b: costlier and worse
        {"id": "d", "mb": 4.0, "acc": 0.90},
        {"id": "e", "mb": 2.0, "acc": 0.40},  # dominated by b: same cost, worse
        {"id": "f", "mb": 5.0, "acc": 0.90},  # dominated by d: costlier, no better
    ]
    assert [p["id"] for p in pareto_frontier(points, "mb", "acc")] == ["a", "b", "d"]


def _pair(gold_intervals, cited):
    return ({"gold": {"cited_intervals": gold_intervals}}, {"cited_intervals": cited})


def test_iou_acceptance_hand_computed():
    pairs = [
        _pair([[0, 10]], [[0, 10]]),  # IoU 1
        _pair([[0, 10]], [[5, 15]]),  # overlap 5, union 15 -> IoU 1/3
        _pair([[0, 10]], []),  # nothing cited -> 0
        ({"gold": {}}, {"cited_intervals": []}),  # no gold intervals: not counted
    ]
    n, accepted = iou_acceptance(pairs, [0.1, 0.5, 1.0])
    assert n == 3
    assert accepted == pytest.approx([2 / 3, 1 / 3, 1 / 3])


def _duration(gold, answer):
    return ({"gold": {"question_type": "duration", "numeric_value": gold}}, {"answer": answer})


def test_tolerance_acceptance_hand_computed():
    pairs = [
        _duration(100, "105 seconds"),  # 5% off
        _duration(100, "120 seconds"),  # 20% off
        _duration(100, "N/A"),  # no number: never accepted
        _duration(0, "0 seconds"),  # exact zero: accepted at any tolerance
    ]
    n, accepted = tolerance_acceptance(pairs, [0.0, 0.10, 0.25])
    assert n == 4
    assert accepted == pytest.approx([1 / 4, 2 / 4, 3 / 4])


def test_counts_are_scored_within_one_bout_not_within_the_duration_tolerance():
    gold = {"question_type": "count", "answer_kind": "numeric", "numeric_value": 5.0}
    assert score_answer({"answer": "6 separate bouts"}, gold)
    assert not score_answer({"answer": "7 separate bouts"}, gold)
    assert not score_answer({"answer": "9 separate bouts"}, gold)


def test_durations_keep_the_pre_registered_tolerance():
    gold = {"question_type": "duration", "answer_kind": "numeric", "numeric_value": 20.0}
    assert score_answer({"answer": "24 seconds"}, gold)  # within the 4 s floor
    assert not score_answer({"answer": "25 seconds"}, gold)
