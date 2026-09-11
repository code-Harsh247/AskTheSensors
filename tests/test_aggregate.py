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


def test_timing_drift_between_minutes_is_not_a_gap():
    """Real recordings are nominally a minute apart but often 61-70 s; the
    unrecorded sliver cannot hide a missing minute, so it is bridged."""
    timeline = build_timeline(minutes([(0, "WALKING"), (61, "WALKING"), (130, "WALKING")]))
    assert spans(timeline) == [("WALKING", 0.0, 190.0)]
    assert timeline.gaps == ()


def test_bridging_stops_at_a_whole_missing_minute():
    # 119 s after the previous start leaves 59 s unclaimed: bridged.
    bridged = build_timeline(minutes([(0, "SITTING"), (119, "SITTING")]))
    assert spans(bridged) == [("SITTING", 0.0, 179.0)]
    # 120 s leaves a full minute unrecorded: a real gap.
    gapped = build_timeline(minutes([(0, "SITTING"), (120, "SITTING")]))
    assert spans(gapped) == [("SITTING", 0.0, 60.0), ("SITTING", 120.0, 180.0)]
    assert gapped.gaps == ((60.0, 120.0),)


def test_drift_before_a_new_activity_belongs_to_the_earlier_minute():
    timeline = build_timeline(minutes([(0, "SITTING"), (75, "WALKING")]))
    assert spans(timeline) == [("SITTING", 0.0, 75.0), ("WALKING", 75.0, 135.0)]


def test_attribution_is_clipped_where_the_next_burst_starts_early():
    """Real example timestamps are not always exactly 60 s apart."""
    timeline = build_timeline(minutes([(0, "SITTING"), (49, "WALKING")]))
    assert spans(timeline) == [("SITTING", 0.0, 49.0), ("WALKING", 49.0, 109.0)]


def test_a_burst_takes_one_label_by_pooled_vote():
    """ExtraSensory labels each minute once, so a label change inside a burst
    cannot be scored; the whole burst takes its pooled majority."""
    windows = [make_window(i, 2.0 * i, "SITTING" if i < 6 else "WALKING") for i in range(10)]
    assert spans(build_timeline(windows)) == [("SITTING", 0.0, 60.0)]


def test_the_pooled_vote_weighs_confidence_not_window_count():
    # 4 confident walking windows outweigh 6 hesitant sitting ones:
    # walking 4*0.9 + 6*(0.7/6) = 4.3 against sitting 6*0.3 + 4*(0.1/6) = 1.87
    windows = [make_window(i, 2.0 * i, "WALKING") for i in range(4)] + [
        make_window(i, 2.0 * i, "SITTING", confidence=0.3) for i in range(4, 10)
    ]
    assert spans(build_timeline(windows)) == [("WALKING", 0.0, 60.0)]


def test_flipped_edge_windows_neither_survive_nor_claim_the_rest_of_the_minute():
    """The failure the oracle-delta dry run exposed: two edge windows flipped
    to different classes tied under smoothing and survived, and a flipped last
    window claimed the unrecorded tail of its minute."""
    windows = [make_window(i, 2.0 * i, "WALKING") for i in range(10)]
    windows[0] = make_window(0, 0.0, "BICYCLING")
    windows[1] = make_window(1, 2.0, "RUNNING")
    windows[9] = make_window(9, 18.0, "STANDING_MOVING")
    assert spans(build_timeline(windows)) == [("WALKING", 0.0, 60.0)]


def test_long_continuous_recordings_keep_window_level_changes():
    """Pooling applies to bursts only; a continuous stretch longer than one
    minute still shows a real change of activity."""
    windows = contiguous(["SITTING"] * 5 + ["WALKING"] * 5)
    assert spans(build_timeline(windows, smoothing_windows=1)) == [
        ("SITTING", 0.0, 50.0),
        ("WALKING", 50.0, 100.0),
    ]


def test_interval_and_gap_lookup():
    timeline = build_timeline(minutes([(0, "WALKING"), (60, "WALKING"), (180, "WALKING")]))

    assert timeline.interval_at(30.0).activity == "WALKING"
    assert timeline.interval_at(150.0) is None
    assert timeline.gap_at(150.0) == (120.0, 180.0)
    assert timeline.interval_at(240.0) is None
    assert timeline.gap_at(240.0) is None
