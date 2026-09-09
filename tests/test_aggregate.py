"""Aggregation is checked against hand-built window sequences whose correct
timeline is obvious by construction."""

import pytest

from ats.aggregate import build_timeline
from ats.contracts import CANONICAL_CLASSES, validate_window_track

WINDOW = 10.0


def window(index: int, t_start: float, activity: str, coverage: float = 1.0):
    confidence = 0.9
    remainder = (1.0 - confidence) / (len(CANONICAL_CLASSES) - 1)
    probs = [remainder] * len(CANONICAL_CLASSES)
    probs[CANONICAL_CLASSES.index(activity)] = confidence
    entry = {
        "window_id": f"w{index:04d}",
        "t_start": t_start,
        "t_end": t_start + WINDOW,
        "probs": probs,
        "coverage": coverage,
        "feature_summary": {
            "acc_mag_mean": 9.8,
            "acc_mag_std": 0.1,
            "dominant_cadence_hz": 0.0,
            "gyro_energy_x": 0.0,
            "gyro_energy_y": 0.0,
            "gyro_energy_z": 0.0,
        },
        "model_id": "test",
    }
    validate_window_track(entry)
    return entry


def sequence(activities, start=0.0, coverage=1.0):
    return [
        window(i, start + i * WINDOW, act, coverage) for i, act in enumerate(activities)
    ]


def test_two_activities_become_two_intervals():
    windows = sequence(["SITTING"] * 5 + ["WALKING"] * 5)
    timeline = build_timeline(windows, smoothing_windows=1)

    assert [(iv.activity, iv.t_start, iv.t_end) for iv in timeline.intervals] == [
        ("SITTING", 0.0, 50.0),
        ("WALKING", 50.0, 100.0),
    ]


def test_smoothing_removes_a_single_window_flicker():
    windows = sequence(["SITTING", "SITTING", "WALKING", "SITTING", "SITTING"])

    unsmoothed = build_timeline(windows, smoothing_windows=1)
    assert len(unsmoothed.intervals) == 3

    smoothed = build_timeline(windows, smoothing_windows=3)
    assert len(smoothed.intervals) == 1
    assert smoothed.intervals[0].activity == "SITTING"
    assert smoothed.intervals[0].t_end == 50.0


def test_gap_breaks_an_interval_even_for_the_same_activity():
    """A duration answer must never silently span a stretch of recording that
    does not exist."""
    windows = sequence(["SITTING"] * 2) + sequence(["SITTING"] * 2, start=100.0)
    timeline = build_timeline(windows, smoothing_windows=1)

    assert len(timeline.intervals) == 2
    assert timeline.intervals[0].as_tuple() == (0.0, 20.0)
    assert timeline.intervals[1].as_tuple() == (100.0, 120.0)
    assert timeline.gaps == ((20.0, 100.0),)
    assert timeline.total_duration("SITTING") == pytest.approx(40.0)


def test_low_coverage_windows_are_dropped():
    windows = sequence(["WALKING"] * 3)
    windows[1]["coverage"] = 0.1
    timeline = build_timeline(windows, smoothing_windows=1, min_coverage=0.5)

    covered = sum(iv.duration for iv in timeline.intervals)
    assert covered == pytest.approx(20.0)
    assert timeline.gaps == ((10.0, 20.0),)


def test_timeline_helpers_on_split_activity():
    windows = sequence(["WALKING"] * 2) + sequence(["WALKING"] * 3, start=100.0)
    timeline = build_timeline(windows, smoothing_windows=1)

    assert timeline.count("WALKING") == 2
    assert timeline.total_duration("WALKING") == pytest.approx(50.0)
    assert timeline.activities_present() == {"WALKING"}
    assert timeline.dominant_activity() == "WALKING"
    assert timeline.span() == (0.0, 130.0)


def test_transitions_report_activity_changes():
    windows = sequence(["SITTING"] * 2 + ["WALKING"] * 2 + ["RUNNING"] * 2)
    timeline = build_timeline(windows, smoothing_windows=1)

    assert timeline.transitions() == [
        (20.0, "SITTING", "WALKING"),
        (40.0, "WALKING", "RUNNING"),
    ]


def test_empty_and_fully_unreliable_tracks_yield_an_empty_timeline():
    assert build_timeline([]).intervals == ()

    windows = sequence(["WALKING"] * 3, coverage=0.0)
    assert build_timeline(windows).intervals == ()


def test_mean_confidence_reflects_assigned_label():
    windows = sequence(["WALKING"] * 3)
    timeline = build_timeline(windows, smoothing_windows=1)
    assert timeline.intervals[0].mean_confidence == pytest.approx(0.9)
