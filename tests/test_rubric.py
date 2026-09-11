"""The explanation rubric (docs/TASKS.md 4B.4): agreement statistics checked
against hand-worked fixtures, and the judge's output checked before it can
become a score. No network: the judge itself is exercised by
scripts/judge_explanations.py."""

import json

import pytest
from _windows import minutes

from ats.eval.rubric import CRITERIA, build_items, judge_prompt, parse_judgment, summarize_runs, weighted_kappa


def test_weighted_kappa_hand_computed():
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    # Every combination of 1 and 2 once: the observed matrix equals the
    # chance matrix, so agreement is exactly chance.
    assert weighted_kappa([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(0.0)
    # Opposite ends every time: disagreement weight 1 on every item, against
    # a chance expectation of 0.5, so kappa = 1 - 1 / 0.5 = -1.
    assert weighted_kappa([1, 5], [5, 1]) == pytest.approx(-1.0)


def test_weighted_kappa_is_undefined_without_variation():
    assert weighted_kappa([4, 4, 4], [4, 4, 4]) is None


def test_weighted_kappa_weighs_near_misses_less():
    near = weighted_kappa([1, 2, 3, 4, 5], [2, 2, 3, 4, 5])
    far = weighted_kappa([1, 2, 3, 4, 5], [5, 2, 3, 4, 5])
    assert 0 < far < near < 1


def _judgment(**scores):
    return json.dumps({**{c: 4 for c in CRITERIA}, "rationale": "ok", **scores})


def test_parse_judgment_accepts_the_scale_and_rejects_anything_else():
    assert parse_judgment(_judgment(cites_real_features=5))["cites_real_features"] == 5
    with pytest.raises(ValueError):
        parse_judgment(_judgment(conclusion_plausible=6))
    with pytest.raises(ValueError):
        parse_judgment(json.dumps({"rationale": "missing scores"}))


def test_summarize_runs():
    run_a = [{"item_id": "x", **{c: 4 for c in CRITERIA}}, {"item_id": "y", **{c: 2 for c in CRITERIA}}]
    run_b = [{"item_id": "x", **{c: 4 for c in CRITERIA}}, {"item_id": "y", **{c: 3 for c in CRITERIA}}]
    report = summarize_runs(run_a, run_b)
    assert report["n_items"] == 2
    first = report["criteria"]["cites_real_features"]
    assert first["mean"] == pytest.approx((4 + 2 + 4 + 3) / 4)
    assert (first["exact_agreement"], first["within_one"]) == (0.5, 1.0)


def test_only_claims_are_judged_and_each_carries_its_measurements():
    windows = minutes([(m, "LYING") for m in (0, 60, 120, 180, 240, 300)])
    questions = [
        {"question_id": "rest", "text": "Was the user resting for a long time?"},
        {"question_id": "nap", "text": "Did the user take a nap?"},
        {"question_id": "car", "text": "Was the user driving?"},  # abstains
        {"question_id": "t1", "text": "What activity is the user performing?"},  # tier 1
    ]
    items, abstained = build_items([("toy", windows, questions)])
    assert [i["item_id"] for i in items] == ["toy:rest", "toy:nap"]
    assert abstained == 1
    assert items[0]["measured"]["still_share"] == pytest.approx(1.0)
    assert '"explanation"' in judge_prompt(items[0])
