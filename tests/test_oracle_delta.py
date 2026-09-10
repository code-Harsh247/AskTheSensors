"""The delta tool must attribute every lost answer to exactly one layer, and
its totals must reconcile."""

import json

import pytest

from ats.eval.delta import attribute, render_markdown, run_delta
from ats.eval.dev import FIXTURES_DIR


@pytest.mark.parametrize(
    "oracle_right, real_right, routed_ok, expected",
    [
        (True, True, True, (None, False)),
        (True, False, True, ("recognition", True)),
        (True, False, False, ("recognition", True)),
        (False, False, True, ("reasoning", True)),
        (False, False, False, ("routing", True)),
        # Wrong on the oracle but right on the real track: a masked bug.
        (False, True, True, ("reasoning", False)),
        (False, True, False, ("routing", False)),
    ],
)
def test_attribution_truth_table(oracle_right, real_right, routed_ok, expected):
    assert attribute(oracle_right, real_right, routed_ok) == expected


def test_identical_tracks_lose_nothing():
    report = run_delta(real_dir=FIXTURES_DIR)
    overall = report["by_question_type"]["overall"]
    assert overall["oracle_right"] == overall["real_right"] == overall["n"]
    assert overall["losses"] == {"recognition": 0, "routing": 0, "reasoning": 0}


@pytest.fixture(scope="module")
def noisy():
    return run_delta(simulate_burst_noise=0.2, seed=0)


def test_isolated_window_flips_are_absorbed_by_the_burst_vote():
    overall = run_delta(simulate_label_noise=0.2, seed=0)["by_question_type"]["overall"]
    assert overall["real_right"] == overall["n"]


def test_noise_on_a_perfect_oracle_is_blamed_only_on_recognition(noisy):
    overall = noisy["by_question_type"]["overall"]
    assert overall["real_right"] < overall["n"], "misclassifying 20% of bursts should cost something"
    assert overall["losses"]["recognition"] == overall["n"] - overall["real_right"]
    assert overall["losses"]["routing"] == overall["losses"]["reasoning"] == 0


def test_every_loss_has_exactly_one_layer(noisy):
    for name, row in noisy["by_question_type"].items():
        assert sum(row["losses"].values()) == row["n"] - row["real_right"], name


def test_simulated_runs_say_so(noisy):
    assert noisy["real_source"].startswith("SIMULATED")
    assert "SIMULATED" in render_markdown(noisy)


def test_markdown_gives_each_member_their_fix_list(noisy):
    markdown = render_markdown(noisy)
    assert "## Fix list - Member A (recognition)" in markdown
    assert "## Fix list - Member B (routing and reasoning)" in markdown
    lost = [r for r in noisy["rows"] if r["layer"] == "recognition"]
    assert f"`{lost[0]['question_id']}`" in markdown


def _one_question_set(tmp_path, question):
    questions_dir = tmp_path / "questions"
    questions_dir.mkdir()
    (questions_dir / "subj_synth_a.json").write_text(json.dumps({"questions": [question]}), encoding="utf-8")
    return questions_dir


def test_a_wrong_operator_is_blamed_on_routing(tmp_path):
    # A count question phrased so the router picks verify.
    questions_dir = _one_question_set(
        tmp_path,
        {
            "question_id": "q",
            "text": "Is the user walking at 610 seconds?",
            "gold": {"answer": "2", "question_type": "count", "answer_kind": "numeric", "numeric_value": 2.0},
        },
    )
    row = run_delta(real_dir=FIXTURES_DIR, questions_dir=questions_dir)["rows"][0]
    assert (row["operator"], row["layer"], row["is_loss"]) == ("verify", "routing", True)


def test_a_wrong_answer_from_a_suitable_operator_is_blamed_on_reasoning(tmp_path):
    # Walking totals 720 s in subj_synth_a; the gold here disagrees.
    questions_dir = _one_question_set(
        tmp_path,
        {
            "question_id": "q",
            "text": "How long was the user walking in total?",
            "gold": {"answer": "999 seconds", "question_type": "duration", "answer_kind": "numeric", "numeric_value": 999.0},
        },
    )
    row = run_delta(real_dir=FIXTURES_DIR, questions_dir=questions_dir)["rows"][0]
    assert (row["operator"], row["layer"]) == ("duration", "reasoning")


def test_a_missing_real_track_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_delta(real_dir=tmp_path)


def test_exactly_one_real_source_is_required():
    with pytest.raises(ValueError):
        run_delta()
    with pytest.raises(ValueError):
        run_delta(real_dir=FIXTURES_DIR, simulate_label_noise=0.1)
    with pytest.raises(ValueError):
        run_delta(simulate_label_noise=0.1, simulate_burst_noise=0.1)
