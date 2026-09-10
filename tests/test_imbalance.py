"""ats/imbalance.py checked against a hand-computed toy class distribution
(docs/TASKS.md testing convention)."""

from __future__ import annotations

import pytest

from ats.imbalance import class_counts, class_weights, sample_weights

CLASSES = ("A", "B", "C")


def test_class_counts_zero_fills_absent_classes():
    labels = ["A", "A", "B"]
    assert class_counts(labels, CLASSES) == {"A": 2, "B": 1, "C": 0}


def test_class_weights_hand_computed():
    # 8 examples: A x4, B x2, C x2. n_total=8, n_classes=3.
    # weight_A = 8 / (3*4) = 0.6667, weight_B = 8/(3*2) = 1.3333, weight_C = 1.3333.
    labels = ["A", "A", "A", "A", "B", "B", "C", "C"]
    weights = class_weights(labels, CLASSES)
    assert weights["A"] == pytest.approx(8 / 12)
    assert weights["B"] == pytest.approx(8 / 6)
    assert weights["C"] == pytest.approx(8 / 6)


def test_class_weights_of_a_perfectly_balanced_set_is_all_ones():
    labels = ["A", "B", "C"] * 5
    weights = class_weights(labels, CLASSES)
    assert weights == {c: pytest.approx(1.0) for c in CLASSES}


def test_class_weights_raises_on_empty_labels():
    with pytest.raises(ValueError):
        class_weights([], CLASSES)


def test_class_weights_raises_on_missing_class():
    labels = ["A", "A", "B"]  # C never occurs
    with pytest.raises(ValueError, match="C"):
        class_weights(labels, CLASSES)


def test_sample_weights_matches_class_weights_per_example():
    labels = ["A", "B", "A", "C"]
    weights = class_weights(labels, CLASSES)
    assert sample_weights(labels, CLASSES) == [weights["A"], weights["B"], weights["A"], weights["C"]]
