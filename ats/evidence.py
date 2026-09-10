"""Evidence attribution (docs/TASKS.md task 2B.3): for a set of cited
intervals, the windows behind them and what those windows actually measured.
Everything an explanation says about the signal comes from here, never from
the language layer.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Sequence

from ats.serialize import format_seconds

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


def windows_in(windows: Sequence[dict[str, Any]], spans: Sequence[Span]) -> list[dict[str, Any]]:
    """Windows whose centre falls inside any span. Using the centre assigns an
    overlapping window to exactly one side of an activity boundary."""
    return [
        w
        for w in windows
        if any(start <= (w["t_start"] + w["t_end"]) / 2.0 < end for start, end in spans)
    ]


def _union_length(spans: Sequence[Span]) -> float:
    total = 0.0
    current: list[float] | None = None
    for start, end in sorted(spans):
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    if current is not None:
        total += current[1] - current[0]
    return total


def summarize(windows: Sequence[dict[str, Any]], spans: Sequence[Span]) -> SignalSummary | None:
    members = windows_in(windows, spans)
    if not members:
        return None
    features = [w["feature_summary"] for w in members]
    cadences = [f["dominant_cadence_hz"] for f in features if f["dominant_cadence_hz"] > 0]
    return SignalSummary(
        n_windows=len(members),
        recorded_s=_union_length([(w["t_start"], w["t_end"]) for w in members]),
        acc_mag_mean=statistics.fmean(f["acc_mag_mean"] for f in features),
        acc_mag_std=statistics.fmean(f["acc_mag_std"] for f in features),
        gyro_energy=statistics.fmean(
            f["gyro_energy_x"] + f["gyro_energy_y"] + f["gyro_energy_z"] for f in features
        ),
        cadence_hz=statistics.median(cadences) if cadences else 0.0,
        cadence_share=len(cadences) / len(members),
    )


def describe(summary: SignalSummary | None) -> str:
    if summary is None:
        return "No windows fall inside the cited intervals."
    text = (
        f"The {summary.n_windows} windows behind this ({format_seconds(summary.recorded_s)} s "
        f"of recorded signal) show accelerometer magnitude averaging {summary.acc_mag_mean:.2f} m/s^2 "
        f"with standard deviation {summary.acc_mag_std:.2f}, and gyroscope energy {summary.gyro_energy:.3g}"
    )
    if summary.cadence_share > 0:
        return text + (
            f"; a periodic cadence near {summary.cadence_hz:.2f} Hz appears in "
            f"{summary.cadence_share:.0%} of windows."
        )
    return text + "; no periodic cadence was detected."
