"""Signal-property predicates (docs/TASKS.md 4B.3): stillness is calibrated
from real data, and open-world answers must be backed by the signal, not only
by the classifier's label."""

import pytest
from _windows import make_window, minutes

from ats.aggregate import build_timeline, load_track
from ats.answer import to_answer
from ats.eval.dev import FIXTURES_DIR
from ats.operators import execute
from ats.routing import route
from ats.signal import STILL_ACC_STD, STILL_GYRO_ENERGY, calibrate_still_thresholds, is_still
from ats.validator import grounding_problems


def ask(windows, text):
    timeline = build_timeline(windows)
    call = route(text)
    answer = to_answer("q", call, execute(call, timeline, windows))
    assert grounding_problems(answer, call, timeline) == []
    return answer


def test_stillness_thresholds_match_their_calibration():
    """The constants are the rounded-up 95th percentiles of lying-down windows
    on subj_real_a, the calibration subject; re-derive them from the track."""
    oracle = load_track(FIXTURES_DIR / "track_subj_real_a.jsonl")
    assert calibrate_still_thresholds(oracle) == (STILL_ACC_STD, STILL_GYRO_ENERGY)


def test_is_still():
    assert is_still(make_window(0, 0.0, "LYING"))
    assert is_still(make_window(0, 0.0, "SITTING"))
    assert not is_still(make_window(0, 0.0, "WALKING"))
    assert is_still(make_window(0, 0.0, "BICYCLING", signal="SITTING"))


def test_a_rest_question_naming_no_posture_is_routed_to_stillness():
    call = route("Was the user resting for a long time?")
    assert (call.op, call.predicate, call.activities) == ("open_world", "prolonged", ())


def test_sustained_stillness_spans_sitting_and_lying():
    """The classifier's sitting-versus-lying confusion does not matter: both
    are still, so the stretch of rest runs across both."""
    windows = minutes([(m, "SITTING") for m in (0, 60, 120)] + [(m, "LYING") for m in (180, 240, 300, 360)])
    answer = ask(windows, "Was the user resting for a long time?")
    assert answer["answer"] == "Likely yes"
    assert answer["activity_event"] == "Sustained stillness"
    assert answer["cited_intervals"] == [[0.0, 180.0], [180.0, 420.0]]
    assert "sitting" in answer["explanation"] and "lying down" in answer["explanation"]


def test_a_moving_minute_breaks_the_stretch():
    windows = minutes(
        [(0, "SITTING"), (60, "SITTING"), (120, "WALKING")] + [(m, "LYING") for m in (180, 240, 300, 360)]
    )
    answer = ask(windows, "Was the user resting for a long time?")
    assert answer["answer"] == "Likely no"
    assert answer["cited_intervals"] == [[180.0, 420.0]]


def test_a_still_signal_labelled_lying_counts_even_if_mislabelled_as_walking():
    windows = minutes([(m, "WALKING", "LYING") for m in (0, 60, 120, 180, 240)])
    assert ask(windows, "Was the user resting for a long time?")["answer"] == "Likely yes"


def test_a_wheeled_claim_needs_a_moving_signal():
    still = minutes([(0, "SITTING"), (60, "BICYCLING", "SITTING"), (120, "SITTING")])
    answer = ask(still, "Was the user using a wheeled or pedal-based mode of movement?")
    assert answer["answer"] == "No"
    assert "mostly still" in answer["explanation"]

    moving = minutes([(0, "SITTING"), (60, "BICYCLING"), (120, "SITTING")])
    answer = ask(moving, "Was the user using a wheeled or pedal-based mode of movement?")
    assert answer["answer"] == "Yes"
    assert answer["cited_intervals"] == [[60.0, 120.0]]


def test_only_the_still_bicycling_intervals_are_set_aside():
    windows = minutes([(0, "BICYCLING", "SITTING"), (120, "SITTING"), (240, "BICYCLING")])
    answer = ask(windows, "Was the user using a wheeled or pedal-based mode of movement?")
    assert answer["answer"] == "Yes"
    assert answer["cited_intervals"] == [[240.0, 300.0]]
    assert "1 bicycling interval(s) with a mostly still signal were set aside" in answer["explanation"]


def _rescaled(windows, factor):
    """The same windows with the accelerometer read in the wrong units."""
    for window in windows:
        features = window["feature_summary"]
        features["acc_mag_mean"] *= factor
        features["acc_mag_std"] *= factor
    return windows


def test_stillness_is_not_judged_when_a_recording_is_far_from_gravity():
    """subj_real_b's accelerometer reads about 9.7 times gravity; against
    thresholds in m/s^2 nothing there looks still, so rest must not be judged."""
    windows = _rescaled(minutes([(m, "LYING") for m in (0, 60, 120, 180, 240, 300)]), 9.81)
    answer = ask(windows, "Was the user resting for a long time?")
    assert answer["answer"] == "N/A"
    assert "gravity" in answer["explanation"]


def test_movement_labels_are_not_vetoed_when_stillness_cannot_be_judged():
    # Read in g instead of m/s^2, a bicycling minute would look still; with the
    # scale off, the label stands rather than being vetoed.
    windows = _rescaled(minutes([(0, "SITTING"), (60, "BICYCLING", "SITTING"), (120, "SITTING")]), 1 / 9.81)
    assert ask(windows, "Was the user using a wheeled or pedal-based mode of movement?")["answer"] == "Yes"


@pytest.mark.parametrize(
    "text",
    ["Was the user doing anything strenuous?", "Was the user doing anything strenuous at 70 seconds?"],
)
def test_a_strenuous_claim_needs_a_moving_signal(text):
    windows = minutes([(0, "SITTING"), (60, "RUNNING", "SITTING"), (120, "SITTING")])
    assert ask(windows, text)["answer"] == "No"
