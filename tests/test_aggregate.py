"""Aggregation is checked against hand-built window sequences whose correct
timeline is obvious by construction."""

import pytest
from _windows import contiguous, make_window, minutes

from ats.aggregate import build_timeline

RECORDED_ONLY = {"attribution_period_s": None}


def spans(timeline):
    return [(iv.activity, iv.t_start, iv.t_end) for iv in timeline.intervals]


# --- Segmentation over the recorded signal ----------------------------------


def test_two_activities_become_two_intervals():
    timeline = build_timeline(
        contiguous(["SITTING"] * 5 + ["WALKING"] * 5), smoothing_windows=1, **RECORDED_ONLY
    )
    assert spans(timeline) == [("SITTING", 0.0, 50.0), ("WALKING", 50.0, 100.0)]


def test_smoothing_removes_a_single_window_flicker():
    windows = contiguous(["SITTING", "SITTING", "WALKING", "SITTING", "SITTING"])

    assert len(build_timeline(windows, smoothing_windows=1, **RECORDED_ONLY).intervals) == 3

    smoothed = build_timeline(windows, smoothing_windows=3, **RECORDED_ONLY)
    assert spans(smoothed) == [("SITTING", 0.0, 50.0)]


def test_gap_breaks_an_interval_even_for_the_same_activity():
    """A duration answer must never silently span a stretch of recording that
    does not exist."""
    windows = contiguous(["SITTING"] * 2) + contiguous(["SITTING"] * 2, start=100.0)
    timeline = build_timeline(windows, smoothing_windows=1, **RECORDED_ONLY)

    assert spans(timeline) == [("SITTING", 0.0, 20.0), ("SITTING", 100.0, 120.0)]
    assert timeline.gaps == ((20.0, 100.0),)
    assert timeline.total_duration("SITTING") == pytest.approx(40.0)


def test_low_coverage_windows_are_dropped():
    windows = contiguous(["WALKING"] * 3)
    windows[1]["coverage"] = 0.1
    timeline = build_timeline(windows, smoothing_windows=1, min_coverage=0.5, **RECORDED_ONLY)

    assert sum(iv.duration for iv in timeline.intervals) == pytest.approx(20.0)
    assert timeline.gaps == ((10.0, 20.0),)


def test_timeline_helpers_on_split_activity():
    windows = contiguous(["WALKING"] * 2) + contiguous(["WALKING"] * 3, start=100.0)
    timeline = build_timeline(windows, smoothing_windows=1, **RECORDED_ONLY)

    assert timeline.count("WALKING") == 2
    assert timeline.total_duration("WALKING") == pytest.approx(50.0)
    assert timeline.activities_present() == {"WALKING"}
    assert timeline.dominant_activity() == "WALKING"
    assert timeline.span() == (0.0, 130.0)


def test_transitions_report_activity_changes():
    windows = contiguous(["SITTING"] * 2 + ["WALKING"] * 2 + ["RUNNING"] * 2)
    timeline = build_timeline(windows, smoothing_windows=1, **RECORDED_ONLY)

    assert timeline.transitions() == [
        (20.0, "SITTING", "WALKING"),
        (40.0, "WALKING", "RUNNING"),
    ]


def test_empty_and_fully_unreliable_tracks_yield_an_empty_timeline():
    assert build_timeline([]).intervals == ()
    assert build_timeline(contiguous(["WALKING"] * 3, coverage=0.0)).intervals == ()


def test_mean_confidence_reflects_assigned_label():
    timeline = build_timeline(contiguous(["WALKING"] * 3), smoothing_windows=1, **RECORDED_ONLY)
    assert timeline.intervals[0].mean_confidence == pytest.approx(0.9)


def test_overlapping_windows_meet_at_the_overlap_midpoint():
    """With 4 s windows at a 2 s hop, ending a run at its last window's end
    would overlap the next run and double-count 2 s at every change."""
    windows = [make_window(i, 2.0 * i, "SITTING" if i < 5 else "WALKING") for i in range(10)]
    timeline = build_timeline(windows, smoothing_windows=1, **RECORDED_ONLY)

    # last sitting window [8, 12], first walking window [10, 14] -> seam at 11
    assert spans(timeline) == [("SITTING", 0.0, 11.0), ("WALKING", 11.0, 22.0)]


# --- Minute attribution ------------------------------------------------------


def test_consecutive_same_activity_minutes_merge_into_one_bout():
    timeline = build_timeline(minutes([(0, "WALKING"), (60, "WALKING"), (120, "WALKING")]))

    assert spans(timeline) == [("WALKING", 0.0, 180.0)]
    assert timeline.gaps == ()


def test_a_missing_minute_is_a_real_gap():
    timeline = build_timeline(minutes([(0, "WALKING"), (60, "WALKING"), (180, "WALKING")]))

    assert spans(timeline) == [("WALKING", 0.0, 120.0), ("WALKING", 180.0, 240.0)]
    assert timeline.gaps == ((120.0, 180.0),)
    assert timeline.count("WALKING") == 2
    assert timeline.total_duration("WALKING") == pytest.approx(180.0)


def test_activity_change_between_minutes():
    timeline = build_timeline(minutes([(0, "SITTING"), (60, "WALKING")]))
    assert spans(timeline) == [("SITTING", 0.0, 60.0), ("WALKING", 60.0, 120.0)]


def test_attribution_is_clipped_where_the_next_burst_starts_early():
    """Real example timestamps are not always exactly 60 s apart."""
    timeline = build_timeline(minutes([(0, "SITTING"), (49, "WALKING")]))
    assert spans(timeline) == [("SITTING", 0.0, 49.0), ("WALKING", 49.0, 109.0)]


def test_activity_change_inside_a_burst_attributes_the_rest_of_the_minute_to_the_last_run():
    windows = [make_window(i, 2.0 * i, "SITTING" if i < 5 else "WALKING") for i in range(10)]
    assert spans(build_timeline(windows)) == [("SITTING", 0.0, 11.0), ("WALKING", 11.0, 60.0)]


def test_interval_and_gap_lookup():
    timeline = build_timeline(minutes([(0, "WALKING"), (60, "WALKING"), (180, "WALKING")]))

    assert timeline.interval_at(30.0).activity == "WALKING"
    assert timeline.interval_at(150.0) is None
    assert timeline.gap_at(150.0) == (120.0, 180.0)
    assert timeline.interval_at(240.0) is None
    assert timeline.gap_at(240.0) is None
