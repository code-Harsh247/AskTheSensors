"""Aggregation layer: per-window predictions -> an activity timeline.

Owned by Member B (docs/TASKS.md task 1B.1). Temporal smoothing and
segmentation live here rather than in the recognition backbone, because the
interval boundaries produced here are exactly what grounding IoU is scored
against.

Minute attribution. ExtraSensory records one ~20 s burst per labeled minute,
and its labels are per minute, so each burst stands for its whole minute: a
contiguous run of windows claims `attribution_period_s` from its first window
(clipped where the next run starts), consecutive same-activity minutes merge
into one bout, and a missing minute stays a real gap. Decided by the team on
2026-09-11 (docs/TASKS.md §0).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ats.contracts import CANONICAL_CLASSES, validate_window_track

DEFAULT_SMOOTHING_WINDOWS = 5
DEFAULT_MIN_COVERAGE = 0.5
EXTRASENSORY_CYCLE_S = 60.0
_EPS = 1e-6


@dataclass(frozen=True)
class Interval:
    activity: str
    t_start: float
    t_end: float
    mean_confidence: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def as_tuple(self) -> tuple[float, float]:
        return (self.t_start, self.t_end)

    def contains(self, t: float) -> bool:
        return self.t_start <= t < self.t_end


@dataclass(frozen=True)
class Timeline:
    """Ordered, non-overlapping activity intervals plus the data gaps between
    them. Gaps are carried explicitly so a duration answer never silently
    spans a stretch of recording that does not exist."""

    intervals: tuple[Interval, ...]
    gaps: tuple[tuple[float, float], ...]

    def intervals_of(self, activity: str) -> list[Interval]:
        return [iv for iv in self.intervals if iv.activity == activity]

    def total_duration(self, activity: str) -> float:
        return sum(iv.duration for iv in self.intervals_of(activity))

    def count(self, activity: str) -> int:
        return len(self.intervals_of(activity))

    def activities_present(self) -> set[str]:
        return {iv.activity for iv in self.intervals}

    def transitions(self) -> list[tuple[float, str, str]]:
        """(time, from_activity, to_activity) for each adjacent interval pair."""
        return [
            (b.t_start, a.activity, b.activity)
            for a, b in zip(self.intervals, self.intervals[1:])
            if a.activity != b.activity
        ]

    def dominant_activity(self) -> str | None:
        if not self.intervals:
            return None
        totals = {act: self.total_duration(act) for act in self.activities_present()}
        return max(totals, key=lambda act: (totals[act], act))

    def span(self) -> tuple[float, float] | None:
        if not self.intervals:
            return None
        return (self.intervals[0].t_start, self.intervals[-1].t_end)

    def interval_at(self, t: float) -> Interval | None:
        return next((iv for iv in self.intervals if iv.contains(t)), None)

    def gap_at(self, t: float) -> tuple[float, float] | None:
        return next((gap for gap in self.gaps if gap[0] <= t < gap[1]), None)


def load_track(path: str | Path) -> list[dict[str, Any]]:
    """Read a window-track JSONL file, validating every record against the
    frozen schema. Raises on the first malformed line."""
    entries: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
            validate_window_track(entry)
            entries.append(entry)
    return entries


def _label_of(entry: dict[str, Any]) -> str:
    probs = entry["probs"]
    return CANONICAL_CLASSES[max(range(len(probs)), key=probs.__getitem__)]


def _infer_gap_tolerance(windows: Sequence[dict[str, Any]]) -> float:
    """Half the median window-to-window step. Derived from the track rather
    than hard-coded, because the window hop is Member A's decision."""
    if len(windows) < 2:
        return 0.0
    steps = [b["t_start"] - a["t_start"] for a, b in zip(windows, windows[1:])]
    positive = [s for s in steps if s > 0]
    if not positive:
        return 0.0
    return statistics.median(positive) / 2.0


def _split_on_gaps(
    windows: Sequence[dict[str, Any]], tolerance: float
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for entry in windows:
        if current and entry["t_start"] - current[-1]["t_end"] > tolerance:
            chunks.append(current)
            current = []
        current.append(entry)
    if current:
        chunks.append(current)
    return chunks


def _mode_smooth(labels: Sequence[str], k: int) -> list[str]:
    if k <= 1 or len(labels) < 2:
        return list(labels)
    half = k // 2
    smoothed: list[str] = []
    for i, own in enumerate(labels):
        window = labels[max(0, i - half) : min(len(labels), i + half + 1)]
        counts = Counter(window)
        top = max(counts.values())
        winners = sorted(label for label, n in counts.items() if n == top)
        # Keep the window's own label when it ties, so smoothing never
        # introduces a flip that the raw predictions did not support.
        smoothed.append(own if own in winners else winners[0])
    return smoothed


def _runs(
    chunk: Sequence[dict[str, Any]], smoothing_windows: int
) -> list[tuple[str, list[dict[str, Any]]]]:
    smoothed = _mode_smooth([_label_of(w) for w in chunk], smoothing_windows)
    runs: list[tuple[str, list[dict[str, Any]]]] = []
    for entry, label in zip(chunk, smoothed):
        if runs and runs[-1][0] == label:
            runs[-1][1].append(entry)
        else:
            runs.append((label, [entry]))
    return runs


def _seam(previous: dict[str, Any], following: dict[str, Any]) -> float:
    """Boundary between two adjacent windows: the midpoint of their overlap.
    With overlapping windows, ending one run at its last window's end would
    make it overlap the next run and double-count that time."""
    return (previous["t_end"] + following["t_start"]) / 2.0


def build_timeline(
    windows: Iterable[dict[str, Any]],
    *,
    smoothing_windows: int = DEFAULT_SMOOTHING_WINDOWS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    gap_tolerance_s: float | None = None,
    attribution_period_s: float | None = EXTRASENSORY_CYCLE_S,
) -> Timeline:
    """Collapse per-window predictions into contiguous activity intervals.

    Windows whose `coverage` falls below `min_coverage` are dropped as
    unreliable, which can itself open a gap. Pass `attribution_period_s=None`
    to keep intervals to the recorded signal only.
    """
    ordered = sorted(windows, key=lambda w: w["t_start"])
    reliable = [w for w in ordered if w["coverage"] >= min_coverage]
    if not reliable:
        return Timeline(intervals=(), gaps=())

    tolerance = _infer_gap_tolerance(ordered) if gap_tolerance_s is None else gap_tolerance_s
    chunks = _split_on_gaps(reliable, tolerance)

    pieces: list[tuple[str, float, float, list[float]]] = []
    gaps: list[tuple[float, float]] = []
    for i, chunk in enumerate(chunks):
        start = chunk[0]["t_start"]
        end = chunk[-1]["t_end"]
        if attribution_period_s is not None:
            end = max(end, start + attribution_period_s)
        if i + 1 < len(chunks):
            next_start = chunks[i + 1][0]["t_start"]
            end = min(end, next_start)
            if next_start - end > _EPS:
                gaps.append((round(end, 3), round(next_start, 3)))

        runs = _runs(chunk, smoothing_windows)
        for j, (label, members) in enumerate(runs):
            run_start = start if j == 0 else _seam(runs[j - 1][1][-1], members[0])
            run_end = end if j == len(runs) - 1 else _seam(members[-1], runs[j + 1][1][0])
            k = CANONICAL_CLASSES.index(label)
            pieces.append((label, run_start, run_end, [w["probs"][k] for w in members]))

    merged: list[list[Any]] = []
    for label, run_start, run_end, confidences in pieces:
        if merged and merged[-1][0] == label and abs(run_start - merged[-1][2]) <= _EPS:
            merged[-1][2] = run_end
            merged[-1][3].extend(confidences)
        else:
            merged.append([label, run_start, run_end, list(confidences)])

    intervals = tuple(
        Interval(
            activity=label,
            t_start=round(run_start, 3),
            t_end=round(run_end, 3),
            mean_confidence=sum(confidences) / len(confidences),
        )
        for label, run_start, run_end, confidences in merged
    )
    return Timeline(intervals=intervals, gaps=tuple(gaps))
