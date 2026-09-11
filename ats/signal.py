"""Signal-property predicates for open-world answers (docs/TASKS.md task
4B.3): claims argued from what the sensors measured, not from the
classifier's label.

Only stillness is calibrated. A window is still when it is as still as 95% of
lying-down windows on subj_real_a, the calibration subject; subj_real_b is
kept aside to check the rule (scripts/calibrate_stillness.py reports both).
subj_real_a has no running or bicycling, so no movement signature such as a
pedalling cadence is calibrated, and no predicate claims one.

The thresholds are in m/s^2, so they are only applied to a recording whose
median acceleration magnitude sits near gravity. The check on subj_real_b
found its accelerometer reading about 9.7 times gravity, a units problem
upstream of this module, and stillness is not judged on such a recording.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from ats.contracts import CANONICAL_CLASSES

# 95th percentiles of lying-down windows on subj_real_a (0.0216 and 0.0047),
# rounded up. tests/test_signal.py re-derives them from the committed track.
STILL_ACC_STD = 0.022
STILL_GYRO_ENERGY = 0.005

# An interval supports a claim about its state only when most of its windows agree.
MAJORITY = 0.5

GRAVITY = 9.81
# A phone's median acceleration magnitude over a recording sits near gravity
# whatever the wearer does. Far outside this band the recording's units are
# wrong -- a factor of 9.81 either way is the usual cause -- and thresholds in
# m/s^2 cannot be applied to it.
GRAVITY_BAND = (8.0, 12.0)


def recording_gravity(windows: Sequence[dict[str, Any]]) -> float | None:
    means = sorted(
        w["feature_summary"]["acc_mag_mean"] for w in windows if w["feature_summary"]["acc_mag_mean"] > 0
    )
    return means[len(means) // 2] if means else None


def gravity_ok(windows: Sequence[dict[str, Any]]) -> bool:
    gravity = recording_gravity(windows)
    return gravity is not None and GRAVITY_BAND[0] <= gravity <= GRAVITY_BAND[1]


def gyro_energy(features: dict[str, float]) -> float:
    return features["gyro_energy_x"] + features["gyro_energy_y"] + features["gyro_energy_z"]


def is_still(window: dict[str, Any]) -> bool:
    features = window["feature_summary"]
    return features["acc_mag_std"] <= STILL_ACC_STD and gyro_energy(features) <= STILL_GYRO_ENERGY


def calibrate_still_thresholds(
    oracle_windows: Sequence[dict[str, Any]], percentile: float = 0.95, min_coverage: float = 0.95
) -> tuple[float, float]:
    """The stillness thresholds implied by an oracle track: the given
    percentile of its well-covered lying-down windows, rounded up to 3
    decimals."""

    def label(window: dict[str, Any]) -> str:
        probs = window["probs"]
        return CANONICAL_CLASSES[max(range(len(probs)), key=probs.__getitem__)]

    lying = [
        w["feature_summary"] for w in oracle_windows if w["coverage"] >= min_coverage and label(w) == "LYING"
    ]
    if not lying:
        raise ValueError("no well-covered lying-down windows to calibrate on")

    def pick(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(percentile * len(ordered)))]

    return (
        math.ceil(pick([f["acc_mag_std"] for f in lying]) * 1000) / 1000,
        math.ceil(pick([gyro_energy(f) for f in lying]) * 1000) / 1000,
    )
