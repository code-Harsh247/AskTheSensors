"""Phase 1 exit criteria as executable checks: the full B-side pipeline runs
end to end on a fixture track and emits 100% schema-valid output for 100% of
questions. Answer *content* is not gated here -- routing lands in Phase 2.
"""

from pathlib import Path

import pytest

from ats.answer import baseline_answer
from ats.aggregate import build_timeline, load_track
from ats.eval import evaluate
from ats.serialize import (
    read_answers_jsonl,
    read_question_set,
    write_answers,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = REPO_ROOT / "data" / "questions_dev.json"
FIXTURES = sorted((REPO_ROOT / "tests" / "fixtures").glob("track_*.jsonl"))


def test_fixture_tracks_exist():
    assert FIXTURES, "run scripts/make_dev_fixture.py to generate fixture tracks"


@pytest.mark.parametrize("track_path", FIXTURES, ids=lambda p: p.stem)
def test_every_question_gets_schema_valid_output(track_path, tmp_path):
    timeline = build_timeline(load_track(track_path))
    question_set = read_question_set(QUESTIONS)
    questions = question_set["questions"]

    answers = [baseline_answer(q["question_id"], timeline) for q in questions]
    out = tmp_path / "ans.jsonl"

    # write_answers and read_answers_jsonl both validate against the frozen
    # schema, so a round trip proves well-formedness in both directions.
    write_answers(answers, out, fmt="jsonl")
    reloaded = read_answers_jsonl(out)

    assert len(reloaded) == len(questions)
    assert {a["question_id"] for a in reloaded} == {q["question_id"] for q in questions}


def test_text_output_round_trips(tmp_path):
    timeline = build_timeline(load_track(FIXTURES[0]))
    question_set = read_question_set(QUESTIONS)
    questions = question_set["questions"]
    answers = [baseline_answer(q["question_id"], timeline) for q in questions]

    out = tmp_path / "ans.txt"
    write_answers(
        answers, out, fmt="text", queries={q["question_id"]: q["text"] for q in questions}
    )
    text = out.read_text(encoding="utf-8")

    assert text.count("Answer:") == len(questions)
    assert text.count("Explanation:") == len(questions)


def test_evaluate_runs_to_completion_and_returns_a_metrics_dict(tmp_path):
    timeline = build_timeline(load_track(FIXTURES[0]))
    question_set = read_question_set(QUESTIONS)
    answers = [
        baseline_answer(q["question_id"], timeline) for q in question_set["questions"]
    ]

    report = evaluate({"answers": answers}, question_set)

    assert report["n_graded"] == len(question_set["questions"])
    assert report["n_missing_predictions"] == 0
    assert 0.0 <= report["overall_macro_accuracy"] <= 1.0
    assert report["iou_threshold"] == 0.5
    assert set(report["by_question_type"]) <= {
        "identification",
        "verification",
        "duration",
        "count",
        "comparison",
        "grounding",
        "open_world",
    }


def test_missing_prediction_counts_as_wrong_not_skipped():
    question_set = read_question_set(QUESTIONS)
    report = evaluate({"answers": []}, question_set)

    assert report["n_graded"] == len(question_set["questions"])
    assert report["n_missing_predictions"] == report["n_graded"]
    assert report["overall_micro_accuracy"] == 0.0


def test_dev_question_set_covers_every_tier_and_edge_case():
    question_set = read_question_set(QUESTIONS)
    golds = [q["gold"] for q in question_set["questions"]]

    tier_of = {
        "identification": 1,
        "verification": 1,
        "duration": 2,
        "count": 2,
        "comparison": 2,
        "grounding": 3,
        "open_world": 4,
    }
    for tier in (1, 2, 3, 4):
        n = sum(1 for g in golds if tier_of[g["question_type"]] == tier)
        assert n >= 12, f"tier {tier} has only {n} questions; TASKS.md requires >= 12"

    answers = [g["answer"] for g in golds]
    assert any(a == "Equal" for a in answers), "missing the comparison-tie edge case"
    assert any(
        g.get("numeric_value") == 0.0 for g in golds
    ), "missing the never-occurring-activity edge case"
    assert any(a == "N/A" for a in answers), "missing the data-gap edge case"
    assert any(
        len(g.get("cited_intervals", [])) > 1 for g in golds
    ), "missing the multi-interval duration edge case"
