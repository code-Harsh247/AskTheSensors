"""End-to-end checks on the dev set: well-formed output on every track, and
the Phase 2 Member B exit criteria (docs/TASKS.md) measured against the
oracle-style fixture tracks."""

import pytest

from ats.aggregate import load_track
from ats.answer import answer_all
from ats.eval import evaluate
from ats.eval.dev import FIXTURES_DIR, QUESTIONS_DIR, dev_subjects, run_dev_eval
from ats.serialize import read_answers_jsonl, read_question_set, write_answers

SUBJECTS = dev_subjects()
ALL_QUESTIONS = [
    q for s in SUBJECTS for q in read_question_set(QUESTIONS_DIR / f"{s}.json")["questions"]
]
TRACKS = sorted(FIXTURES_DIR.glob("track_*.jsonl"))


def test_every_dev_subject_has_a_fixture_track():
    assert len(SUBJECTS) >= 3
    for subject in SUBJECTS:
        assert (FIXTURES_DIR / f"track_{subject}.jsonl").exists()


@pytest.mark.parametrize("track_path", TRACKS, ids=lambda p: p.stem)
def test_every_question_gets_schema_valid_output(track_path, tmp_path):
    answers, _ = answer_all(ALL_QUESTIONS, load_track(track_path))
    out = tmp_path / "ans.jsonl"

    # write_answers and read_answers_jsonl both validate against the frozen
    # schema, so a round trip proves well-formedness in both directions.
    write_answers(answers, out, fmt="jsonl")
    reloaded = read_answers_jsonl(out)

    assert [a["question_id"] for a in reloaded] == [q["question_id"] for q in ALL_QUESTIONS]


def test_text_output_round_trips(tmp_path):
    subject = SUBJECTS[0]
    questions = read_question_set(QUESTIONS_DIR / f"{subject}.json")["questions"]
    answers, _ = answer_all(questions, load_track(FIXTURES_DIR / f"track_{subject}.jsonl"))

    out = tmp_path / "ans.txt"
    write_answers(answers, out, fmt="text", queries={q["question_id"]: q["text"] for q in questions})
    text = out.read_text(encoding="utf-8")

    assert text.count("Answer:") == len(questions)
    assert text.count("Explanation:") == len(questions)


def test_missing_prediction_counts_as_wrong_not_skipped():
    report = evaluate({"answers": []}, {"questions": ALL_QUESTIONS})
    assert report["n_graded"] == len(ALL_QUESTIONS)
    assert report["n_missing_predictions"] == report["n_graded"]
    assert report["overall_micro_accuracy"] == 0.0


def test_dev_question_set_covers_every_tier_and_edge_case():
    golds = [q["gold"] for q in ALL_QUESTIONS]
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
    assert "Equal" in answers, "missing the comparison-tie edge case"
    assert any(g.get("numeric_value") == 0.0 for g in golds), "missing the never-occurring-activity edge case"
    assert "N/A" in answers, "missing the data-gap edge case"
    assert any(
        g["question_type"] == "duration" and len(g.get("cited_intervals", [])) > 1 for g in golds
    ), "missing the multi-interval duration edge case"


# --- Phase 2 exit criteria ---------------------------------------------------


@pytest.fixture(scope="module")
def oracle_report():
    return run_dev_eval()


def test_phase2_tier1_and_tier2_accuracy(oracle_report):
    assert oracle_report["by_tier"]["1"]["accuracy"] >= 0.95
    assert oracle_report["by_tier"]["2"]["accuracy"] >= 0.95


def test_phase2_grounded_accuracy(oracle_report):
    assert oracle_report["grounded_accuracy"] >= 0.90


def test_no_answer_is_withheld_on_a_clean_oracle(oracle_report):
    """With perfect labels every computed answer is grounded; a rejection here
    would be a bug in an operator, not a property of the data."""
    assert oracle_report["n_rejected_by_validator"] == 0


@pytest.mark.parametrize("noise", [{"label_noise": 0.1}, {"burst_noise": 0.1}], ids=["windows", "bursts"])
def test_reasoning_degrades_gracefully_under_noise(noise):
    report = run_dev_eval(seed=0, **noise)
    assert report["n_graded"] == len(ALL_QUESTIONS)
    assert report["n_missing_predictions"] == 0
    assert report["n_empty_answers"] == 0
