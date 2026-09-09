"""Aggregation layer: per-window predictions -> an activity timeline.

Owned by Member B (docs/TASKS.md task 1B.1). Temporal smoothing and
segmentation live here rather than in the recognition backbone, because the
interval boundaries produced here are exactly what grounding IoU is scored
against.
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


def _label_of(entry: dict[str, Any]) -> tuple[str, float]:
    probs = entry["probs"]
    best = max(range(len(probs)), key=lambda i: probs[i])
    return CANONICAL_CLASSES[best], probs[best]


def _infer_gap_tolerance(windows: Sequence[dict[str, Any]]) -> float:
    """Half the median window-to-window step. Derived from the track rather
    than hard-coded, because the window hop is Member A's Phase 1 decision."""
    if len(windows) < 2:
        return 0.0
    steps = [b["t_start"] - a["t_start"] for a, b in zip(windows, windows[1:])]
    positive = [s for s in steps if s > 0]
    if not positive:
        return 0.0
    return statistics.median(positive) / 2.0


def _split_on_gaps(
    windows: Sequence[dict[str, Any]], tolerance: float
) -> tuple[list[list[dict[str, Any]]], list[tuple[float, float]]]:
    chunks: list[list[dict[str, Any]]] = []
    gaps: list[tuple[float, float]] = []
    current: list[dict[str, Any]] = []
    for entry in windows:
        if current:
            previous_end = current[-1]["t_end"]
            if entry["t_start"] - previous_end > tolerance:
                gaps.append((previous_end, entry["t_start"]))
                chunks.append(current)
                current = []
        current.append(entry)
    if current:
        chunks.append(current)
    return chunks, gaps


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


def build_timeline(
    windows: Iterable[dict[str, Any]],
    *,
    smoothing_windows: int = DEFAULT_SMOOTHING_WINDOWS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    gap_tolerance_s: float | None = None,
) -> Timeline:
    """Collapse per-window predictions into contiguous activity intervals.

    Windows whose `coverage` falls below `min_coverage` are dropped as
    unreliable, which can itself open a gap. Segmentation never spans a gap.
    """
    ordered = sorted(windows, key=lambda w: w["t_start"])
    reliable = [w for w in ordered if w["coverage"] >= min_coverage]
    if not reliable:
        return Timeline(intervals=(), gaps=())

    tolerance = _infer_gap_tolerance(ordered) if gap_tolerance_s is None else gap_tolerance_s
    chunks, gaps = _split_on_gaps(reliable, tolerance)

    intervals: list[Interval] = []
    for chunk in chunks:
        labelled = [_label_of(w) for w in chunk]
        smoothed = _mode_smooth([label for label, _ in labelled], smoothing_windows)

        run_start = 0
        for i in range(1, len(chunk) + 1):
            if i < len(chunk) and smoothed[i] == smoothed[run_start]:
                continue
            members = chunk[run_start:i]
            label = smoothed[run_start]
            confidences = [
                w["probs"][CANONICAL_CLASSES.index(label)] for w in members
            ]
            intervals.append(
                Interval(
                    activity=label,
                    t_start=members[0]["t_start"],
                    t_end=members[-1]["t_end"],
                    mean_confidence=sum(confidences) / len(confidences),
                )
            )
            run_start = i

    return Timeline(intervals=tuple(intervals), gaps=tuple(gaps))
