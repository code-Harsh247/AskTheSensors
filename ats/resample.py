"""Resample per-channel raw samples onto an exact 25 Hz grid, with an
explicit gap policy: gaps beyond `MAX_INTERP_GAP_S` between real samples are
marked missing rather than interpolated (docs/TASKS.md 1A.3, 1A.4; PRD Sec
3.2 hard requirement).

Also anchors a subject's separately-timestamped accelerometer/gyroscope
bursts onto one shared "seconds from start of recording" timeline (the
frozen time base, docs/TASKS.md Sec 0) -- see `globalize_subject`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ats.ingest import Burst, Subject

TARGET_HZ = 25.0

# Gaps between consecutive real samples up to this long are bridged by linear
# interpolation; longer gaps leave every grid point strictly between them
# marked missing (PRD Sec 3.2, docs/TASKS.md 1A.4). Chosen well above the
# largest jitter seen within one ~20s recording burst (single-digit
# milliseconds) and well below the ~40s dead time between bursts, so it
# cleanly separates "sensor hiccup" from "no recording happened here" without
# ever bridging across a burst boundary.
MAX_INTERP_GAP_S = 1.0


@dataclass(frozen=True)
class GlobalSamples:
    """One subject's accelerometer and gyroscope samples, all on the shared
    seconds-from-start-of-recording timeline, plus the per-burst ground-truth
    label spans on that same timeline."""

    acc: tuple[tuple[float, float, float, float], ...]  # (t, x, y, z), sorted by t
    gyro: tuple[tuple[float, float, float, float], ...]
    label_spans: tuple[tuple[str | None, float, float], ...]  # (activity, t_start, t_end)
    t_end: float


def globalize_subject(subject: Subject) -> GlobalSamples:
    """Anchor every burst onto one timeline. A burst's own device clock is
    accurate for *spacing between samples within that burst*, but says
    nothing about wall-clock alignment across bursts, so each burst's first
    sample is anchored to its `example_ts` (the per-minute key shared with
    the labels file) and later samples keep their measured offset from it.
    This is a documented approximation -- the true 20-second recording could
    start anywhere within its labeled minute -- and is stated as such in the
    report rather than presented as exact.
    """
    if not subject.bursts:
        raise ValueError(f"subject {subject.subject_id!r} has no usable bursts")

    t0 = subject.bursts[0].example_ts
    acc: list[tuple[float, float, float, float]] = []
    gyro: list[tuple[float, float, float, float]] = []
    spans: list[tuple[str | None, float, float]] = []

    for burst in subject.bursts:
        burst_start = burst.example_ts - t0
        starts = [burst_start]
        ends = [burst_start]

        if burst.acc:
            local0 = burst.acc[0][0]
            for t, x, y, z in burst.acc:
                acc.append((burst_start + (t - local0), x, y, z))
            ends.append(burst_start + (burst.acc[-1][0] - local0))

        if burst.gyro:
            local0 = burst.gyro[0][0]
            for t, x, y, z in burst.gyro:
                gyro.append((burst_start + (t - local0), x, y, z))
            ends.append(burst_start + (burst.gyro[-1][0] - local0))

        spans.append((burst.activity, min(starts), max(ends)))

    acc.sort(key=lambda row: row[0])
    gyro.sort(key=lambda row: row[0])
    # `[x] * bool(acc)` looks like a guard but isn't one -- `acc[-1]` is
    # evaluated unconditionally before the multiplication, so it crashes on
    # an empty list regardless. A real subject can have every burst missing
    # one whole channel (e.g. gyroscope unavailable on their phone for the
    # entire recording), so acc or gyro being empty here isn't hypothetical.
    candidates = [spans[-1][2]]
    if acc:
        candidates.append(acc[-1][0])
    if gyro:
        candidates.append(gyro[-1][0])
    t_end = max(candidates)
    return GlobalSamples(tuple(acc), tuple(gyro), tuple(spans), t_end)


def _value_at(g: float, left: tuple[float, float] | None, right: tuple[float, float] | None) -> float | None:
    if left is None or right is None:
        return None  # no extrapolation before the first or after the last real sample
    span = right[0] - left[0]
    if span > MAX_INTERP_GAP_S:
        return None  # gap between the bracketing real samples too large to bridge
    if span == 0:
        return left[1]
    frac = (g - left[0]) / span
    return left[1] + frac * (right[1] - left[1])


def resample_channel(
    samples: Sequence[tuple[float, float]],
    t_end: float,
    t_start: float = 0.0,
    hz: float = TARGET_HZ,
) -> tuple[tuple[float, ...], tuple[float | None, ...]]:
    """Resample one scalar channel onto a uniform grid at exactly `hz`.

    `samples` must be sorted, jittered timestamps are expected. Returns
    `(times, values)`: `times` is an exact arithmetic sequence (so it is
    trivially free of duplicate or backward steps -- PRD Sec 3.2's hard
    25 Hz requirement), and `values[i]` is `None` wherever the nearest real
    samples bracketing `times[i]` are more than `MAX_INTERP_GAP_S` apart.
    """
    n = max(0, round((t_end - t_start) * hz))
    times = tuple(t_start + i / hz for i in range(n + 1))
    if not samples:
        return times, tuple(None for _ in times)

    ts = [s[0] for s in samples]
    vs = [s[1] for s in samples]
    values: list[float | None] = []
    j = 0
    for g in times:
        while j < len(ts) and ts[j] < g:
            j += 1
        if j < len(ts) and ts[j] == g:
            values.append(vs[j])
            continue
        left = (ts[j - 1], vs[j - 1]) if j > 0 else None
        right = (ts[j], vs[j]) if j < len(ts) else None
        values.append(_value_at(g, left, right))
    return times, tuple(values)
