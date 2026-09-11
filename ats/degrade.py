"""Signal-quality degradation injectors for the robustness curve
(docs/TASKS.md task 5A.2, PRD Sec 7.4.5): additive noise at a stated SNR,
sample dropping at a stated fraction, and decimation below 25 Hz.

Operates on `ats.resample.GlobalSamples`, *before* windowing, so the same
degraded signal flows into both `feature_summary` and the CNN's input
tensor consistently -- degrading only the model's tensor would leave
`feature_summary` (and therefore every explanation) describing the clean
signal, which would not be a real end-to-end robustness test.

Decimation's "then upsampled" half (PRD Sec 7.4.5) needs no separate step:
`ats.resample.resample_channel` already interpolates whatever real samples
remain back onto the fixed 25 Hz grid, for any subject, always -- thinning
the real samples first and letting that existing interpolation fill the
gaps *is* decimate-then-upsample.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ats.resample import GlobalSamples

Row = tuple[float, float, float, float]

DEGRADATION_KINDS = ("noise", "dropout", "decimate")


def _add_noise(rows: Sequence[Row], snr_db: float, rng: np.random.Generator) -> tuple[Row, ...]:
    if not rows:
        return ()
    times = [r[0] for r in rows]
    xyz = np.array([[x, y, z] for _, x, y, z in rows])
    signal_power = float(np.mean(xyz**2))
    if signal_power == 0:
        return tuple(rows)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noisy = xyz + rng.normal(0.0, math.sqrt(noise_power), size=xyz.shape)
    return tuple((t, *row) for t, row in zip(times, noisy.tolist()))


def _drop_samples(rows: Sequence[Row], fraction: float, rng: np.random.Generator) -> tuple[Row, ...]:
    if not rows:
        return ()
    keep = rng.random(len(rows)) >= fraction
    return tuple(r for r, k in zip(rows, keep) if k)


def _decimate(rows: Sequence[Row], factor: int) -> tuple[Row, ...]:
    """Keeps every `factor`-th real sample by order, simulating a lower
    device sample rate."""
    if factor <= 1:
        return tuple(rows)
    return tuple(rows[i] for i in range(0, len(rows), factor))


def degrade_global_samples(g: GlobalSamples, kind: str, level: float, seed: int = 0) -> GlobalSamples:
    """`kind='noise'`: `level` is the target SNR in dB (lower = noisier).
    `kind='dropout'`: `level` is the fraction of samples dropped (0-1).
    `kind='decimate'`: `level` is the integer keep-every-Nth-sample factor
    (>=1; 1 means no degradation, matching a "none" baseline row)."""
    if kind not in DEGRADATION_KINDS:
        raise ValueError(f"unknown degradation kind {kind!r}, expected one of {DEGRADATION_KINDS}")
    rng = np.random.default_rng(seed)
    if kind == "noise":
        acc, gyro = _add_noise(g.acc, level, rng), _add_noise(g.gyro, level, rng)
    elif kind == "dropout":
        acc, gyro = _drop_samples(g.acc, level, rng), _drop_samples(g.gyro, level, rng)
    else:
        acc, gyro = _decimate(g.acc, int(level)), _decimate(g.gyro, int(level))
    return GlobalSamples(acc=acc, gyro=gyro, label_spans=g.label_spans, t_end=g.t_end)
