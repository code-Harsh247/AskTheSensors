"""Exit criterion for docs/TASKS.md Phase 1, Member A: subject-wise splits
never leak a subject across train/val/test."""

from __future__ import annotations

import json

import pytest

from ats.splits import load_splits, make_splits, save_splits


def _all_subjects(n: int) -> list[str]:
    return [f"subj_{i:03d}" for i in range(n)]


@pytest.mark.parametrize("n,seed", [(10, 0), (37, 1), (60, 2), (5, 3)])
def test_zero_subject_overlap_across_splits(n, seed):
    subjects = _all_subjects(n)
    splits = make_splits(subjects, seed=seed)

    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(subjects)


def test_splits_are_deterministic_given_the_same_seed():
    subjects = _all_subjects(20)
    a = make_splits(subjects, seed=7)
    b = make_splits(subjects, seed=7)
    assert a == b


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        make_splits(_all_subjects(10), ratios=(0.5, 0.5, 0.5))


def test_duplicate_subject_ids_rejected():
    with pytest.raises(ValueError):
        make_splits(["a", "a", "b"])


def test_save_and_load_round_trip(tmp_path):
    subjects = _all_subjects(15)
    splits = make_splits(subjects, seed=4)
    path = tmp_path / "splits.json"
    save_splits(splits, path)

    loaded = load_splits(path)
    assert loaded == splits

    with path.open() as f:
        raw = json.load(f)
    assert set(raw) == {"train", "val", "test"}
