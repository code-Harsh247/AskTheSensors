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
from ats.signal import (
    NO_FLOOR,
    STILL_ACC_STD,
    STILL_GYRO_ENERGY,
    calibrate_still_thresholds,
    gyro_floor,
    is_still,
)
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
    """Before the units fix in ats/ingest.py, subj_real_b's accelerometer read
    about 9.7 times gravity; against thresholds in m/s^2 nothing there looks
    still, so rest must not be judged."""
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


Y_OFFSET = 0.02  # the size of subj_real_b's y-axis resting energy


def _gyro_offset(windows, y_offset=Y_OFFSET):
    """The same windows from a phone whose gyroscope y-axis carries a constant
    bias, adding the same energy to every window."""
    for window in windows:
        window["feature_summary"]["gyro_energy_y"] += y_offset
    return windows


def test_the_resting_floor_is_read_from_accelerometer_quiet_windows():
    # 20 lying windows (quiet) and 10 walking (not): the floor is the lying
    # windows' per-axis energy, 0.0005 on x and z, 0.0005 + 0.02 on y.
    windows = _gyro_offset(minutes([(0, "LYING"), (60, "LYING"), (120, "WALKING")]))
    assert gyro_floor(windows) == pytest.approx((0.0005, 0.0205, 0.0005))


def test_no_floor_without_enough_quiet_windows():
    # 9 quiet windows is one short of MIN_QUIET_WINDOWS; nothing is subtracted.
    windows = minutes([(0, "WALKING"), (60, "RUNNING")])
    assert gyro_floor(windows) == NO_FLOOR
    windows[:9] = minutes([(0, "LYING")])[:9]
    assert gyro_floor(windows) == NO_FLOOR


def test_a_gyroscope_offset_does_not_hide_stillness():
    windows = _gyro_offset(minutes([(m, "LYING") for m in (0, 60, 120, 180, 240, 300)]))
    assert not is_still(windows[0])  # 0.0215 raw energy, over the 0.005 threshold
    assert is_still(windows[0], gyro_floor(windows))
    assert ask(windows, "Was the user resting for a long time?")["answer"] == "Likely yes"


def test_the_floor_is_scaled_by_coverage():
    # A half-covered window carries half the offset's energy; subtracting the
    # full floor would push it below zero and call a moving axis still.
    floor = (0.0, 0.02, 0.0)
    moving = {"coverage": 0.5, "feature_summary": {"acc_mag_std": 0.01, "gyro_energy_x": 0.0,
                                                   "gyro_energy_y": 0.016, "gyro_energy_z": 0.0}}
    assert not is_still(moving, floor)  # 0.016 - 0.5 * 0.02 = 0.006, over 0.005


def test_restlessness_is_movement_while_the_posture_stays_at_rest():
    # Sitting for three minutes, but two of them carry a moving signal: 20 of
    # 30 resting windows are not still, a majority.
    windows = minutes([(0, "SITTING", "STANDING_MOVING"), (60, "SITTING", "STANDING_MOVING"), (120, "SITTING")])
    answer = ask(windows, "Was the user fidgeting?")
    assert answer["answer"] == "Likely yes"
    assert answer["cited_intervals"] == [[0.0, 180.0]]
    assert "67% of the 30 windows" in answer["explanation"]


def test_a_still_rest_is_not_restless_and_cites_every_resting_interval():
    windows = minutes([(0, "SITTING"), (60, "WALKING"), (120, "LYING"), (180, "LYING")])
    answer = ask(windows, "Was the user restless?")
    assert answer["answer"] == "Likely no"
    assert answer["cited_intervals"] == [[0.0, 60.0], [120.0, 240.0]]  # the walking minute is not rest


def test_restlessness_is_not_judged_without_any_rest():
    windows = minutes([(0, "WALKING"), (60, "RUNNING")])
    assert ask(windows, "Was the user fidgeting?")["answer"] == "N/A"


def test_sleep_is_argued_from_stillness_and_says_what_it_cannot_tell():
    windows = minutes([(m, "LYING") for m in (0, 60, 120, 180, 240, 300)])
    answer = ask(windows, "Did the user take a nap?")
    assert answer["answer"] == "Likely yes"
    assert answer["activity_event"] == "Sustained stillness, consistent with sleep"
    assert "cannot tell sleep from resting awake" in answer["explanation"]


@pytest.mark.parametrize(
    "text",
    ["Was the user driving?", "Was the user climbing stairs?", "Was the user dancing?", "Was the user on a bus?"],
)
def test_behaviours_with_no_calibrated_signature_are_not_guessed(text):
    """No signal signature for these is calibrated, so the honest answer is an
    explicit abstention, never the nearest of the seven labels."""
    windows = minutes([(0, "SITTING"), (60, "BICYCLING"), (120, "WALKING")])
    answer = ask(windows, text)
    assert (answer["answer"], answer["cited_intervals"]) == ("N/A", [])


def test_an_offset_does_not_make_movement_look_still():
    windows = _gyro_offset(minutes([(0, "SITTING"), (60, "SITTING"), (120, "BICYCLING"), (180, "SITTING")]))
    answer = ask(windows, "Was the user using a wheeled or pedal-based mode of movement?")
    assert answer["answer"] == "Yes"
    assert answer["cited_intervals"] == [[120.0, 180.0]]
