"""Evidence attribution (docs/TASKS.md task 2B.3): for a set of cited
intervals, the windows behind them and what those windows actually measured.
Everything an explanation says about the signal comes from here, never from
the language layer.
"""

from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

from ats.serialize import format_seconds
from ats.signal import GyroFloor, gyro_floor, is_still

# The recognition backbone consumes all six channels (ats/model.py
# INPUT_CHANNELS), so any claim derived from its labels rests on both
# modalities and every channel.
MODALITY = "both"
MODALITY_TEXT = "Accelerometer, Gyroscope"
CHANNELS = ("all",)
CHANNELS_TEXT = "All"

Span = tuple[float, float]


@dataclass(frozen=True)
class SignalSummary:
    n_windows: int
    recorded_s: float
    acc_mag_mean: float
    acc_mag_std: float
    gyro_energy: float
    cadence_hz: float
    cadence_share: float
    still_share: float


def _centre(window: dict[str, Any]) -> float:
    return (window["t_start"] + window["t_end"]) / 2.0


def _merge(spans: Sequence[Span]) -> list[Span]:
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def windows_in(windows: Sequence[dict[str, Any]], spans: Sequence[Span]) -> list[dict[str, Any]]:
    """Windows whose centre falls inside any span. Using the centre assigns an
    overlapping window to exactly one side of an activity boundary. Binary
    search keeps this fast on real recordings with thousands of bouts."""
    merged = _merge(spans)
    starts = [start for start, _ in merged]
    members: list[dict[str, Any]] = []
    for window in windows:
        centre = _centre(window)
        i = bisect.bisect_right(starts, centre) - 1
        if i >= 0 and centre < merged[i][1]:
            members.append(window)
    return members


def _summary_of(members: Sequence[dict[str, Any]], floor: GyroFloor) -> SignalSummary | None:
    if not members:
        return None
    features = [w["feature_summary"] for w in members]
    cadences = [f["dominant_cadence_hz"] for f in features if f["dominant_cadence_hz"] > 0]
    return SignalSummary(
        n_windows=len(members),
        recorded_s=sum(end - start for start, end in _merge([(w["t_start"], w["t_end"]) for w in members])),
        acc_mag_mean=statistics.fmean(f["acc_mag_mean"] for f in features),
        acc_mag_std=statistics.fmean(f["acc_mag_std"] for f in features),
        gyro_energy=statistics.fmean(
            f["gyro_energy_x"] + f["gyro_energy_y"] + f["gyro_energy_z"] for f in features
        ),
        cadence_hz=statistics.median(cadences) if cadences else 0.0,
        cadence_share=len(cadences) / len(members),
        still_share=sum(1 for w in members if is_still(w, floor)) / len(members),
    )


def summarize(windows: Sequence[dict[str, Any]], spans: Sequence[Span]) -> SignalSummary | None:
    """The gyroscope's resting floor comes from the whole recording, not just
    the cited windows, so a cited stretch of rest cannot hide its own offset."""
    return _summary_of(windows_in(windows, spans), gyro_floor(windows))


def summarize_each(windows: Sequence[dict[str, Any]], spans: Sequence[Span]) -> list[SignalSummary | None]:
    """One summary per span, in a single pass over the windows. Spans must not
    overlap, which holds for timeline intervals."""
    floor = gyro_floor(windows)
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    starts = [spans[i][0] for i in order]
    buckets: list[list[dict[str, Any]]] = [[] for _ in spans]
    for window in windows:
        centre = _centre(window)
        k = bisect.bisect_right(starts, centre) - 1
        if k >= 0 and centre < spans[order[k]][1]:
            buckets[order[k]].append(window)
    return [_summary_of(bucket, floor) for bucket in buckets]


def describe(summary: SignalSummary | None) -> str:
    if summary is None:
        return "No windows fall inside the cited intervals."
    text = (
        f"The {summary.n_windows} windows behind this ({format_seconds(summary.recorded_s)} s "
        f"of recorded signal) show accelerometer magnitude averaging {summary.acc_mag_mean:.2f} m/s^2 "
        f"with standard deviation {summary.acc_mag_std:.2f}, and gyroscope energy {summary.gyro_energy:.3g}"
    )
    if summary.cadence_share > 0:
        text += (
            f"; a periodic cadence near {summary.cadence_hz:.2f} Hz appears in "
            f"{summary.cadence_share:.0%} of windows."
        )
    else:
        text += "; no periodic cadence was detected."
    return text + f" {summary.still_share:.0%} of these windows show a still signal."
