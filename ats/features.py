"""Time- and frequency-domain feature extraction per window (docs/TASKS.md
task 2A.1), for training/evaluating the recognition backbone.

Distinct from `ats/windowing.py:feature_summary` -- that function computes
the small, frozen `window_track.feature_summary` set Member B cites verbatim
in explanations (the A<->B contract; do not extend it here). This module
computes a much richer feature vector for internal use by the classifier
and by the two required Phase 2 baselines, and is entirely Member A's own
concern.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np  # see docs/CITATIONS.md#numpy-python-library

from ats.windowing import TARGET_HZ, Window, _dominant_cadence_hz

# Frequency bands (Hz) the spectral-energy features are pooled over. Chosen
# to span DC/near-static through the fastest motion a 25Hz stream can
# resolve (Nyquist = 12.5Hz), coarse enough to be meaningful at a 4s/100-
# sample window (0.25Hz bin resolution).
SPECTRAL_BANDS: tuple[tuple[float, float], ...] = ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 12.5))

_ACC_AXES = ("acc_x", "acc_y", "acc_z")
_GYRO_AXES = ("gyro_x", "gyro_y", "gyro_z")


def _clean_axis(rows: Sequence[tuple], axis: int) -> list[float]:
    return [row[axis] for row in rows if row[axis] is not None]


def _clean_triplets(rows: Sequence[tuple]) -> list[tuple[float, float, float]]:
    return [row for row in rows if all(v is not None for v in row)]


def _stats(values: Sequence[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "rms": 0.0}
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    rms = math.sqrt(sum(v * v for v in values) / n)
    return {"mean": mean, "std": math.sqrt(variance), "min": min(values), "max": max(values), "rms": rms}


def _jerk_stats(values: Sequence[float], hz: float) -> dict[str, float]:
    """Rate of change of the signal (m/s^3 for acceleration), from simple
    finite differences at the resample rate."""
    if len(values) < 2:
        return {"mean": 0.0, "std": 0.0}
    jerk = [(b - a) * hz for a, b in zip(values, values[1:])]
    n = len(jerk)
    mean = sum(jerk) / n
    variance = sum((v - mean) ** 2 for v in jerk) / n
    return {"mean": mean, "std": math.sqrt(variance)}


def _spectral_energy_bands(values: Sequence[float], hz: float) -> list[float]:
    """Power summed into each of SPECTRAL_BANDS via a real FFT (numpy). This
    runs twice per window (acc and gyro magnitude) across every window in a
    subject's recording, so a vectorized transform rather than a hand-rolled
    O(n^2) DFT is what keeps building the feature dataset tractable."""
    n = len(values)
    if n < 4:
        return [0.0] * len(SPECTRAL_BANDS)
    arr = np.asarray(values, dtype=float)
    centered = arr - arr.mean()
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(n, d=1.0 / hz)
    power = spectrum.real**2 + spectrum.imag**2

    band_energy = []
    for lo, hi in SPECTRAL_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        band_energy.append(float(power[mask].sum()))
    return band_energy


def _correlation(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    denom = math.sqrt(var_a * var_b)
    return cov / denom if denom > 0 else 0.0


def compute_features(window: Window, hz: float = TARGET_HZ) -> dict[str, float]:
    """A fixed-order, fixed-key feature vector per window: per-channel
    time-domain stats (including magnitude), jerk, spectral energy bands,
    dominant cadence, and cross-axis correlation -- PRD's magnitude
    statistics / cadence / spectral energy / jerk / axis-correlation list
    (docs/TASKS.md 2A.1), computed only from real or interpolated signal
    (never from a gap)."""
    features: dict[str, float] = {}

    acc_triplets = _clean_triplets(window.acc)
    gyro_triplets = _clean_triplets(window.gyro)
    acc_mag = [math.sqrt(x * x + y * y + z * z) for x, y, z in acc_triplets]
    gyro_mag = [math.sqrt(x * x + y * y + z * z) for x, y, z in gyro_triplets]

    for name, axis in zip(_ACC_AXES, range(3)):
        values = _clean_axis(window.acc, axis)
        for stat, value in _stats(values).items():
            features[f"{name}_{stat}"] = value
        for stat, value in _jerk_stats(values, hz).items():
            features[f"{name}_jerk_{stat}"] = value

    for stat, value in _stats(acc_mag).items():
        features[f"acc_mag_{stat}"] = value
    for stat, value in _jerk_stats(acc_mag, hz).items():
        features[f"acc_mag_jerk_{stat}"] = value
    for band_idx, energy in enumerate(_spectral_energy_bands(acc_mag, hz)):
        features[f"acc_mag_spectral_energy_band{band_idx}"] = energy
    features["dominant_cadence_hz"] = _dominant_cadence_hz(acc_mag, hz) if acc_mag else 0.0

    for name, axis in zip(_GYRO_AXES, range(3)):
        values = _clean_axis(window.gyro, axis)
        for stat, value in _stats(values).items():
            features[f"{name}_{stat}"] = value

    for stat, value in _stats(gyro_mag).items():
        features[f"gyro_mag_{stat}"] = value
    for band_idx, energy in enumerate(_spectral_energy_bands(gyro_mag, hz)):
        features[f"gyro_mag_spectral_energy_band{band_idx}"] = energy

    if acc_triplets:
        ax = [row[0] for row in acc_triplets]
        ay = [row[1] for row in acc_triplets]
        az = [row[2] for row in acc_triplets]
        features["acc_corr_xy"] = _correlation(ax, ay)
        features["acc_corr_xz"] = _correlation(ax, az)
        features["acc_corr_yz"] = _correlation(ay, az)
    else:
        features["acc_corr_xy"] = features["acc_corr_xz"] = features["acc_corr_yz"] = 0.0

    if gyro_triplets:
        gx = [row[0] for row in gyro_triplets]
        gy = [row[1] for row in gyro_triplets]
        gz = [row[2] for row in gyro_triplets]
        features["gyro_corr_xy"] = _correlation(gx, gy)
        features["gyro_corr_xz"] = _correlation(gx, gz)
        features["gyro_corr_yz"] = _correlation(gy, gz)
    else:
        features["gyro_corr_xy"] = features["gyro_corr_xz"] = features["gyro_corr_yz"] = 0.0

    return features


FEATURE_NAMES: tuple[str, ...] = tuple(
    compute_features(
        Window(
            t_start=0.0,
            t_end=4.0,
            acc=((0.0, 0.0, 9.8),) * 4,
            gyro=((0.0, 0.0, 0.0),) * 4,
            coverage=1.0,
        )
    ).keys()
)
"""The fixed, ordered set of feature column names `compute_features` always
returns -- used to build a consistent feature matrix across windows."""
