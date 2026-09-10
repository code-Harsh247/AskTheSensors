"""Fixed-length, fixed-hop windowing over resampled channels, plus the
per-window `feature_summary` values the window_track schema requires.

Window length and hop (docs/TASKS.md task 1A.5, frozen 2026-09-10):
    WINDOW_LENGTH_S = 4.0
    HOP_S           = 2.0   (50% overlap)

Rationale, since TASKS.md calls this decision out as needing justification:
- Must fit inside one ~20-second recording burst with room to spare -- a
  window can never span the ~40-second dead time between bursts (aggregation
  never segments across a gap; see ats/aggregate.py), so anything close to
  20s was never in consideration.
- 4 seconds covers several gait cycles even at walking cadence (~1.5-2.5Hz),
  which is enough to estimate `dominant_cadence_hz` without the window being
  so short that a single mis-timed step dominates it.
- 2-second hop keeps the duration tolerance PRD Sec 3.2 derives from it,
  `max(2 x hop, 10% relative)` = `max(4s, 10%)`, tight enough that grounding
  IoU scoring isn't flattered by a coarse floor.
This is the number Member B's `DURATION_ABS_TOL_S` (ats/eval/metrics.py) has
been waiting on since Phase 1 (see docs/TASKS.md Sec 0).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np  # see docs/CITATIONS.md#numpy-python-library

from ats.resample import TARGET_HZ, GlobalSamples, resample_channel

WINDOW_LENGTH_S = 4.0
HOP_S = 2.0

# A window is treated as having no clear periodic motion (sedentary
# postures) unless some frequency bin in the gait band clearly stands out
# above the others -- a peak-to-mean-power ratio, not an absolute energy
# floor, since even a near-stationary window has nonzero magnitude variance
# from ordinary accelerometer noise, and that noise's total energy scales
# with window length/rate rather than with whether real periodic motion is
# present. Threshold chosen by sweeping ratio values against a real
# ExtraSensory subject's labeled windows (see docs/CITATIONS.md#extrasensory-raw-file-layout):
# 8.0 keeps sedentary (SITTING/LYING/STANDING) false-positive rates under 3%
# while still flagging ~20% of WALKING windows -- lower ratios (e.g. an
# absolute-energy-only floor) let broadband sensor noise alias into a
# spurious cadence on a near-motionless window.
_CADENCE_BAND_HZ = (0.5, 4.0)
_CADENCE_PEAK_RATIO = 8.0


@dataclass(frozen=True)
class Window:
    t_start: float
    t_end: float
    acc: tuple[tuple[float | None, float | None, float | None], ...]  # per-sample (x, y, z)
    gyro: tuple[tuple[float | None, float | None, float | None], ...]
    coverage: float  # fraction of the 6 channels' samples in-window that are non-missing


def _slice_span(samples, ts: list[float], t_start: float, t_end: float):
    """samples (with precomputed timestamp index `ts`, both sorted by t) ->
    just the samples inside [t_start, t_end], in O(log n) per call. `ts` is
    built once per subject and reused across every burst; rebuilding it per
    call would make windowing quadratic in the number of bursts."""
    lo = bisect.bisect_left(ts, t_start)
    hi = bisect.bisect_right(ts, t_end)
    return samples[lo:hi]


def _channel_axes(samples, t_start: float, t_end: float, hz: float):
    axes = []
    for axis in range(3):
        scalar = [(t, row[axis]) for t, *row in samples]
        _, values = resample_channel(scalar, t_end, t_start=t_start, hz=hz)
        axes.append(values)
    return list(zip(*axes))


def make_windows(
    globalized: GlobalSamples,
    window_s: float = WINDOW_LENGTH_S,
    hop_s: float = HOP_S,
    hz: float = TARGET_HZ,
):
    """Slide a fixed window/hop within each recording burst, yielding one
    `Window` at a time.

    A generator, not a list: a subject with a full multi-day recording can
    have tens of thousands of windows (one real subject produced over
    60,000), and materializing all of them into a list before a caller even
    starts processing the first one was enough to cause out-of-memory
    crashes both locally and on Kaggle. Every real caller (ats/oracle.py,
    scripts/build_feature_dataset.py) only ever makes one pass, consuming
    and discarding each window as it goes, so nothing here needs the whole
    list at once; wrap in `list(...)` at the call site if you genuinely do
    (e.g. a test asserting non-emptiness).

    Windows are generated per burst span (`globalized.label_spans`), never
    across the whole subject timeline: a subject's bursts are scattered
    across a multi-day span with mostly dead time between them (only ~20s of
    every ~60s is ever recorded), so tiling blindly across [0, t_end] would
    generate orders of magnitude more empty windows than real ones -- and,
    since a window can never legitimately cross a burst boundary anyway (see
    ats/oracle.py), those windows would all be discarded downstream regardless.
    Each burst's channels are resampled once and sliced into overlapping
    windows by index, rather than re-resampling per window.
    """
    points_per_window = round(window_s * hz) + 1
    points_per_hop = round(hop_s * hz)
    if points_per_window <= 0 or points_per_hop <= 0:
        raise ValueError("window_s and hop_s must be positive multiples of 1/hz")

    acc_ts = [s[0] for s in globalized.acc]
    gyro_ts = [s[0] for s in globalized.gyro]

    for _activity, span_start, span_end in globalized.label_spans:
        if span_end - span_start < window_s:
            continue
        local_acc = _slice_span(globalized.acc, acc_ts, span_start, span_end)
        local_gyro = _slice_span(globalized.gyro, gyro_ts, span_start, span_end)
        acc_axes = _channel_axes(local_acc, span_start, span_end, hz)
        gyro_axes = _channel_axes(local_gyro, span_start, span_end, hz)
        n_points = len(acc_axes)

        start_idx = 0
        while start_idx + points_per_window <= n_points:
            acc_slice = tuple(acc_axes[start_idx : start_idx + points_per_window])
            gyro_slice = tuple(gyro_axes[start_idx : start_idx + points_per_window])
            t_start = span_start + start_idx / hz
            t_end = t_start + window_s

            total = len(acc_slice) * 3 + len(gyro_slice) * 3
            present = sum(1 for row in acc_slice for v in row if v is not None) + sum(
                1 for row in gyro_slice for v in row if v is not None
            )
            coverage = present / total if total else 0.0

            yield Window(t_start=t_start, t_end=t_end, acc=acc_slice, gyro=gyro_slice, coverage=coverage)
            start_idx += points_per_hop


def label_for_window(t_start: float, t_end: float, spans) -> str | None:
    """A window's ground truth is the activity of the burst it falls inside.
    Windows are shorter than a burst by construction (WINDOW_LENGTH_S well
    under one ~20s burst), so a window that straddles two spans, or falls
    entirely in the dead time between bursts, has no single ground truth and
    is skipped rather than guessed. Shared by ats/oracle.py and
    scripts/build_feature_dataset.py, both of which need real windows paired
    with their ground-truth label."""
    for activity, span_start, span_end in spans:
        if span_start <= t_start and t_end <= span_end:
            return activity
    return None


def _dominant_cadence_hz(acc_mag: list[float], hz: float) -> float:
    """Peak frequency in the gait band, via a real FFT (numpy, not a
    hand-rolled O(n^2) DFT -- this runs once per window across every window
    in a subject's recording, so the vectorized transform matters)."""
    n = len(acc_mag)
    if n < 4:
        return 0.0
    arr = np.asarray(acc_mag, dtype=float)
    centered = arr - arr.mean()
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(n, d=1.0 / hz)
    power = spectrum.real**2 + spectrum.imag**2

    band_mask = (freqs >= _CADENCE_BAND_HZ[0]) & (freqs <= _CADENCE_BAND_HZ[1])
    band_power = power[band_mask]
    band_freqs = freqs[band_mask]
    if band_power.size == 0:
        return 0.0

    best_idx = int(np.argmax(band_power))
    best_power = float(band_power[best_idx])
    mean_power = float(band_power.mean())
    if mean_power == 0.0 or best_power < _CADENCE_PEAK_RATIO * mean_power:
        return 0.0  # no bin stands out from broadband noise -- not a real cadence
    return float(band_freqs[best_idx])


def feature_summary(window: Window, hz: float = TARGET_HZ) -> dict[str, float]:
    """The window_track schema's frozen `feature_summary` keys. Values are
    computed from real (or interpolated) signal wherever available; a window
    with zero coverage reports all-zero, defensible-but-meaningless numbers
    -- its `coverage` field is what tells downstream code not to trust it.
    """
    mags = [
        math.sqrt(x * x + y * y + z * z)
        for x, y, z in window.acc
        if x is not None and y is not None and z is not None
    ]
    if mags:
        mean = sum(mags) / len(mags)
        variance = sum((m - mean) ** 2 for m in mags) / len(mags)
        acc_mag_mean, acc_mag_std = mean, math.sqrt(variance)
        dominant_cadence_hz = _dominant_cadence_hz(mags, hz)
    else:
        acc_mag_mean = acc_mag_std = dominant_cadence_hz = 0.0

    gyro_energy = []
    for axis in range(3):
        values = [row[axis] for row in window.gyro if row[axis] is not None]
        gyro_energy.append(sum(v * v for v in values))

    return {
        "acc_mag_mean": acc_mag_mean,
        "acc_mag_std": acc_mag_std,
        "dominant_cadence_hz": dominant_cadence_hz,
        "gyro_energy_x": gyro_energy[0],
        "gyro_energy_y": gyro_energy[1],
        "gyro_energy_z": gyro_energy[2],
    }
