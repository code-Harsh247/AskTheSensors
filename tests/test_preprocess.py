"""Exit criteria for docs/TASKS.md Phase 1, Member A: resampling to exactly
25 Hz, gait-band content surviving resampling, and gap coverage landing on
exactly the windows it should.
"""

from __future__ import annotations

import math
import random

from ats.ingest import Burst, Subject
from ats.resample import MAX_INTERP_GAP_S, TARGET_HZ, GlobalSamples, globalize_subject, resample_channel
from ats.windowing import HOP_S, WINDOW_LENGTH_S, make_windows


def _jittered_timestamps(t_end: float, nominal_hz: float, jitter_frac: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    dt = 1.0 / nominal_hz
    ts = []
    t = 0.0
    while t < t_end:
        ts.append(t)
        t += dt * (1.0 + rng.uniform(-jitter_frac, jitter_frac))
    return ts


def test_resample_grid_is_exact_25hz_with_no_duplicate_or_backward_steps():
    ts = _jittered_timestamps(t_end=20.0, nominal_hz=34.0, jitter_frac=0.3, seed=1)
    samples = [(t, math.sin(t)) for t in ts]

    times, _values = resample_channel(samples, t_end=20.0)

    assert times[0] == 0.0
    deltas = [b - a for a, b in zip(times, times[1:])]
    assert all(abs(d - 1.0 / TARGET_HZ) < 1e-9 for d in deltas), "grid spacing must be exactly 1/25 s"
    assert all(d > 0 for d in deltas), "grid must be strictly increasing: no duplicate or backward steps"


def test_1_5hz_sinusoid_survives_resampling_within_tolerance():
    freq = 1.5
    t_end = 20.0
    ts = _jittered_timestamps(t_end=t_end, nominal_hz=34.0, jitter_frac=0.3, seed=2)
    samples = [(t, math.sin(2 * math.pi * freq * t)) for t in ts]

    times, values = resample_channel(samples, t_end=t_end)

    # Drop the leading/trailing edge (no extrapolation before the first or
    # after the last real sample -- see ats/resample.py) before comparing.
    interior = [(t, v) for t, v in zip(times, values) if v is not None]
    assert len(interior) > 0.9 * len(times), "resampling should cover almost the whole interior"
    max_error = max(abs(v - math.sin(2 * math.pi * freq * t)) for t, v in interior)
    assert max_error < 0.05, f"1.5Hz sinusoid distorted too much by resampling: max error {max_error}"


def test_injected_gap_produces_coverage_below_one_on_exactly_the_overlapping_windows():
    hz = TARGET_HZ
    t_end = 40.0

    # A clean, fully-covered accelerometer stream at ~34Hz...
    ts = _jittered_timestamps(t_end=t_end, nominal_hz=34.0, jitter_frac=0.1, seed=3)
    # ...with a real gap punched out well beyond the interpolatable threshold.
    gap_start, gap_end = 20.0, 26.0
    assert gap_end - gap_start > MAX_INTERP_GAP_S
    ts = [t for t in ts if not (gap_start < t < gap_end)]
    acc = tuple((t, math.sin(t), 0.0, 9.81) for t in ts)
    gyro = tuple((t, 0.0, 0.0, 0.0) for t in ts)  # gyro fully covered, no gap

    # End the recording exactly at the last real sample, so the terminal grid
    # point has real data to resolve to rather than needing extrapolation
    # past it -- that edge effect is a separate, expected property of
    # resampling (no extrapolation) and not what this test is checking.
    globalized = GlobalSamples(acc=acc, gyro=gyro, label_spans=(("SITTING", 0.0, ts[-1]),), t_end=ts[-1])
    windows = make_windows(globalized, window_s=WINDOW_LENGTH_S, hop_s=HOP_S, hz=hz)

    def overlaps_gap(w) -> bool:
        # Each window's resampled grid includes both endpoints (see
        # ats/resample.py), so adjacent overlapping windows (hop < window
        # length) share their boundary sample -- treat the window as the
        # closed interval [t_start, t_end] when checking overlap.
        return w.t_start <= gap_end and w.t_end >= gap_start

    for w in windows:
        if overlaps_gap(w):
            assert w.coverage < 1.0, f"window [{w.t_start},{w.t_end}) overlaps the gap but reports full coverage"
        else:
            assert w.coverage == 1.0, f"window [{w.t_start},{w.t_end}) does not overlap the gap but lost coverage"


def test_globalize_subject_handles_a_channel_entirely_missing_across_every_burst():
    """A real subject can have every burst missing one whole channel (e.g.
    gyroscope unavailable on their phone for the entire recording) -- this
    crashed globalize_subject with an IndexError until fixed (a `* bool(...)`
    guard doesn't short-circuit; `acc[-1]` was evaluated before the
    multiplication regardless)."""
    bursts = tuple(
        Burst(example_ts=1000.0 + i * 60.0, acc=(), gyro=((500.0, 0.0, 0.0, 0.0), (500.04, 0.0, 0.0, 0.0)), activity="SITTING")
        for i in range(3)
    )
    subject = Subject(subject_id="no-acc-subject", bursts=bursts)

    globalized = globalize_subject(subject)

    assert globalized.acc == ()
    assert len(globalized.gyro) == 6  # 2 samples x 3 bursts
    assert globalized.t_end > 0.0
