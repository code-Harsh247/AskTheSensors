"""ats/features.py checked against hand-computed values, not just against
itself (docs/TASKS.md testing convention: metric/feature implementations
need fixtures worked out by hand)."""

from __future__ import annotations

import math

import pytest

from ats.features import (
    FEATURE_NAMES,
    _correlation,
    _jerk_stats,
    _spectral_energy_bands,
    _stats,
    compute_features,
)
from ats.windowing import Window


def test_stats_on_a_hand_computed_ramp():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = _stats(values)
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["std"] == pytest.approx(math.sqrt(2.0))  # population std of 1..5
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["rms"] == pytest.approx(math.sqrt(11.0))  # sqrt((1+4+9+16+25)/5)


def test_stats_on_empty_input_is_all_zero():
    assert _stats([]) == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "rms": 0.0}


def test_jerk_stats_on_a_unit_ramp():
    # A ramp with unit steps at hz=1 has constant jerk of 1.0 and zero spread.
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    jerk = _jerk_stats(values, hz=1.0)
    assert jerk["mean"] == pytest.approx(1.0)
    assert jerk["std"] == pytest.approx(0.0)

    # Scaling the sample rate scales jerk proportionally.
    jerk_25hz = _jerk_stats(values, hz=25.0)
    assert jerk_25hz["mean"] == pytest.approx(25.0)


def test_correlation_perfect_positive_negative_and_none():
    a = [1.0, 2.0, 3.0, 4.0]
    b_positive = [2.0, 4.0, 6.0, 8.0]  # b = 2a: perfectly correlated
    b_negative = [4.0, 3.0, 2.0, 1.0]  # perfectly anti-correlated
    b_constant = [5.0, 5.0, 5.0, 5.0]  # zero variance: correlation undefined -> 0.0

    assert _correlation(a, b_positive) == pytest.approx(1.0)
    assert _correlation(a, b_negative) == pytest.approx(-1.0)
    assert _correlation(a, b_constant) == 0.0


def test_spectral_energy_concentrates_in_the_expected_band():
    hz = 25.0
    n = 100
    # A pure 2Hz tone: energy should land in the [1,3)Hz band and be
    # negligible in the others.
    values = [math.sin(2 * math.pi * 2.0 * i / hz) for i in range(n)]
    bands = _spectral_energy_bands(values, hz)
    loudest = bands.index(max(bands))
    assert loudest == 1  # SPECTRAL_BANDS[1] == (1.0, 3.0)
    assert bands[1] > 10 * sum(b for i, b in enumerate(bands) if i != 1)


def test_spectral_energy_of_flat_signal_is_zero():
    assert _spectral_energy_bands([5.0] * 50, hz=25.0) == [0.0, 0.0, 0.0, 0.0]


def test_compute_features_returns_the_fixed_feature_names():
    window = Window(
        t_start=0.0,
        t_end=4.0,
        acc=((0.1, 0.2, 9.8), (0.1, 0.2, 9.8), (0.1, 0.2, 9.8), (0.1, 0.2, 9.8)),
        gyro=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        coverage=1.0,
    )
    features = compute_features(window)
    assert set(features) == set(FEATURE_NAMES)
    assert features["acc_x_mean"] == pytest.approx(0.1)
    assert features["acc_z_mean"] == pytest.approx(9.8)
    assert features["acc_x_std"] == pytest.approx(0.0)  # constant signal


def test_compute_features_handles_gaps_without_crashing():
    window = Window(
        t_start=0.0,
        t_end=4.0,
        acc=((0.1, 0.2, 9.8), (None, None, None), (0.1, 0.2, 9.8), (None, None, None)),
        gyro=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (None, None, None), (None, None, None)),
        coverage=0.5,
    )
    features = compute_features(window)
    assert set(features) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features.values())
