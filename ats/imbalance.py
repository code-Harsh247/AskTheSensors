"""Class-imbalance handling for the recognition backbone (docs/TASKS.md task
2A.3). PRD Sec 3.2 flags the sedentary skew (lying/sitting/standing dominate
walking/running/bicycling) as part of the problem, not an excuse, so the
policy is explicit and testable rather than left to whatever a training
library defaults to.

Policy: inverse-frequency ("balanced") class weights, computed once from the
*training split's* label counts and passed to the classifier's loss --
matches the standard convention (e.g. sklearn's `class_weight="balanced"`):

    weight_c = n_total / (n_classes * count_c)

so a class weight of 1.0 means "exactly at the average frequency", and rarer
classes get proportionally larger weight. This is the mechanism; the written
justification (docs/TASKS.md 2A.3 asks for one) belongs in the report once
real cross-subject class counts are available from
`results/raw/window_features.csv` -- a policy chosen before seeing that
distribution would be a guess, not a justification.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def class_counts(labels: Sequence[str], classes: Sequence[str]) -> dict[str, int]:
    """Per-class counts in `classes` order, zero-filled for classes absent
    from `labels` (a class missing from the training split entirely is a
    real finding, not something to hide by omitting it)."""
    counter = Counter(labels)
    return {c: counter.get(c, 0) for c in classes}


def class_weights(labels: Sequence[str], classes: Sequence[str]) -> dict[str, float]:
    """Inverse-frequency class weights: weight_c = n_total / (n_classes * count_c).

    Raises if `labels` is empty or if any class in `classes` never occurs --
    a zero-count class produces an undefined (infinite) weight, and silently
    dropping it would hide exactly the kind of imbalance this function
    exists to surface.
    """
    if not labels:
        raise ValueError("labels is empty; class weights over zero examples are undefined")
    counts = class_counts(labels, classes)
    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        raise ValueError(
            f"class(es) {missing} never occur in labels; class weights are undefined for them "
            "-- widen the training data or drop the class explicitly, don't silently zero-weight it"
        )
    n_total = len(labels)
    n_classes = len(classes)
    return {c: n_total / (n_classes * n) for c, n in counts.items()}


def sample_weights(labels: Sequence[str], classes: Sequence[str]) -> list[float]:
    """Per-example weights (one per label in `labels`, same order), for
    training APIs that want a sample_weight array rather than a per-class
    dict."""
    weights = class_weights(labels, classes)
    return [weights[label] for label in labels]
