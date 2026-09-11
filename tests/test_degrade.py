"""ats/degrade.py (docs/TASKS.md task 5A.2): degradation injectors for the
robustness curve. Noise/dropout are checked statistically (large-N, a
tolerance band) since they're randomized by construction; decimation is
exact and hand-computable."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ats.degrade import degrade_global_samples
from ats.resample import GlobalSamples


def _constant_rows(n: int, value: float = 1.0, hz: float = 25.0) -> tuple:
    return tuple((i / hz, value, value, value) for i in range(n))


def _make_global(acc, gyro=()) -> GlobalSamples:
    return GlobalSamples(acc=acc, gyro=gyro, label_spans=(("SITTING", 0.0, 10.0),), t_end=10.0)


def test_noise_hits_the_target_snr_approximately():
    n = 20000
    rows = _constant_rows(n, value=2.0)
    g = _make_global(rows)
    target_snr_db = 10.0

    degraded = degrade_global_samples(g, "noise", target_snr_db, seed=0)

    # Matches ats.degrade._add_noise's own definition: mean(x^2+y^2+z^2 per
    # element averaged over all elements), i.e. value^2 here, not summed
    # across the three axes.
    signal_power = 2.0**2
    noise = np.array([[x - 2.0, y - 2.0, z - 2.0] for _, x, y, z in degraded.acc])
    measured_noise_power = float(np.mean(noise**2))
    measured_snr_db = 10 * math.log10(signal_power / measured_noise_power)
    assert measured_snr_db == pytest.approx(target_snr_db, abs=1.0)


def test_noise_leaves_timestamps_and_count_unchanged():
    rows = _constant_rows(100)
    g = _make_global(rows)
    degraded = degrade_global_samples(g, "noise", 20.0, seed=0)
    assert len(degraded.acc) == len(rows)
    assert [r[0] for r in degraded.acc] == [r[0] for r in rows]


def test_dropout_removes_approximately_the_right_fraction():
    n = 20000
    rows = _constant_rows(n)
    g = _make_global(rows)

    degraded = degrade_global_samples(g, "dropout", 0.3, seed=0)

    kept_fraction = len(degraded.acc) / n
    assert kept_fraction == pytest.approx(0.7, abs=0.02)


def test_dropout_only_removes_rows_never_alters_kept_values():
    rows = _constant_rows(500, value=3.0)
    g = _make_global(rows)
    degraded = degrade_global_samples(g, "dropout", 0.5, seed=0)
    assert set(degraded.acc) <= set(rows)


def test_decimate_keeps_exactly_every_nth_sample():
    rows = _constant_rows(10)
    g = _make_global(rows)

    degraded = degrade_global_samples(g, "decimate", 3, seed=0)

    assert degraded.acc == (rows[0], rows[3], rows[6], rows[9])


def test_decimate_factor_of_one_is_a_no_op():
    rows = _constant_rows(10)
    g = _make_global(rows)
    degraded = degrade_global_samples(g, "decimate", 1, seed=0)
    assert degraded.acc == rows


def test_degrade_preserves_label_spans_and_t_end():
    rows = _constant_rows(50)
    g = _make_global(rows)
    for kind, level in (("noise", 15.0), ("dropout", 0.2), ("decimate", 2)):
        degraded = degrade_global_samples(g, kind, level, seed=0)
        assert degraded.label_spans == g.label_spans
        assert degraded.t_end == g.t_end


def test_empty_channel_stays_empty():
    g = _make_global(acc=_constant_rows(10), gyro=())
    for kind, level in (("noise", 10.0), ("dropout", 0.5), ("decimate", 2)):
        degraded = degrade_global_samples(g, kind, level, seed=0)
        assert degraded.gyro == ()


def test_unknown_kind_raises():
    g = _make_global(_constant_rows(10))
    with pytest.raises(ValueError):
        degrade_global_samples(g, "bogus", 1.0)
