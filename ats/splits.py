"""Subject-wise train/val/test splits (docs/TASKS.md task 1A.6).

Splitting by window or example instead of by subject leaks the same person's
gait/posture signature across train and test and inflates every downstream
number -- this module's only job is to guarantee that never happens.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DEFAULT_RATIOS = (0.6, 0.2, 0.2)  # train, val, test


def make_splits(
    subject_ids: list[str],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Deterministically partition `subject_ids` into train/val/test with no
    subject appearing in more than one split."""
    train_r, val_r, test_r = ratios
    if abs(train_r + val_r + test_r - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("duplicate subject_ids")

    ordered = sorted(subject_ids)  # deterministic starting order before shuffling
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = round(n * train_r)
    n_val = round(n * val_r)
    # Remainder (if any, from rounding) goes to test so every subject lands
    # in exactly one split.
    train_ids = ordered[:n_train]
    val_ids = ordered[n_train : n_train + n_val]
    test_ids = ordered[n_train + n_val :]
    return {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}


def save_splits(splits: dict[str, list[str]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, sort_keys=True)
        f.write("\n")


def load_splits(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
