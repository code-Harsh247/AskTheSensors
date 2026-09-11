"""Signal-property predicates for open-world answers (docs/TASKS.md task
4B.3): claims argued from what the sensors measured, not from the
classifier's label.

Only stillness is calibrated. A window is still when it is as still as 95% of
lying-down windows on subj_real_a, the calibration subject;
scripts/calibrate_stillness.py also reports subj_real_b. subj_real_a has no
running or bicycling, so no movement signature such as a pedalling cadence is
calibrated, and no predicate claims one.

Gyroscope energy is measured above each recording's resting floor. A phone's
gyroscope can carry a constant offset (bias) that the processed data does not
remove: subj_real_b reads about 0.014 rad/s on one axis whether its wearer
lies, sits or stands, which adds a fixed amount of energy to every window and
alone kept all of its windows over the threshold. A constant offset b over n
samples adds n*b^2 to a window's energy (the cross term averages out with the
zero-mean motion), so it is estimated per axis from the windows whose
accelerometer is quiet, and subtracted. This correction was designed after
the subj_real_b check had failed, so subj_real_b no longer tests the gyroscope
half of the rule independently.

The thresholds are in m/s^2, so they are only applied to a recording whose
median acceleration magnitude sits near gravity: far outside that band the
recording's units are wrong, and stillness is not judged on it.
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

GYRO_AXES = ("gyro_energy_x", "gyro_energy_y", "gyro_energy_z")
# The resting floor is the 5th percentile of each axis's energy over
# well-covered windows whose accelerometer is quiet (std within
# STILL_ACC_STD). A recording with fewer quiet windows than this has no
# resting stretch to estimate an offset from, and nothing is subtracted.
FLOOR_PERCENTILE = 0.05
MIN_QUIET_WINDOWS = 10
NO_FLOOR = (0.0, 0.0, 0.0)

GyroFloor = tuple[float, float, float]


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


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(percentile * len(ordered)))]


def gyro_floor(windows: Sequence[dict[str, Any]], min_coverage: float = 0.95) -> GyroFloor:
    """Each gyroscope axis's resting energy in this recording: the offset
    every window carries even when the wearer is still."""
    quiet = [
        w["feature_summary"]
        for w in windows
        if w["coverage"] >= min_coverage and w["feature_summary"]["acc_mag_std"] <= STILL_ACC_STD
    ]
    if len(quiet) < MIN_QUIET_WINDOWS:
        return NO_FLOOR
    x, y, z = (_percentile([f[axis] for f in quiet], FLOOR_PERCENTILE) for axis in GYRO_AXES)
    return (x, y, z)


def gyro_energy_above_floor(window: dict[str, Any], floor: GyroFloor = NO_FLOOR) -> float:
    """Gyroscope energy with the resting floor removed. Energy is summed over
    a window's samples, so the floor is scaled by the window's coverage."""
    features = window["feature_summary"]
    return sum(
        max(0.0, features[axis] - level * window["coverage"]) for axis, level in zip(GYRO_AXES, floor)
    )


def is_still(window: dict[str, Any], floor: GyroFloor = NO_FLOOR) -> bool:
    return (
        window["feature_summary"]["acc_mag_std"] <= STILL_ACC_STD
        and gyro_energy_above_floor(window, floor) <= STILL_GYRO_ENERGY
    )


def calibrate_still_thresholds(
    oracle_windows: Sequence[dict[str, Any]], percentile: float = 0.95, min_coverage: float = 0.95
) -> tuple[float, float]:
    """The stillness thresholds implied by an oracle track: the given
    percentile of its well-covered lying-down windows, rounded up to 3
    decimals. Gyroscope energy is taken above the track's resting floor,
    as is_still judges it."""

    def label(window: dict[str, Any]) -> str:
        probs = window["probs"]
        return CANONICAL_CLASSES[max(range(len(probs)), key=probs.__getitem__)]

    lying = [w for w in oracle_windows if w["coverage"] >= min_coverage and label(w) == "LYING"]
    if not lying:
        raise ValueError("no well-covered lying-down windows to calibrate on")
    floor = gyro_floor(oracle_windows, min_coverage)

    return (
        math.ceil(_percentile([w["feature_summary"]["acc_mag_std"] for w in lying], percentile) * 1000) / 1000,
        math.ceil(_percentile([gyro_energy_above_floor(w, floor) for w in lying], percentile) * 1000) / 1000,
    )
