"""Generate the frozen subject-wise train/val/test split (docs/TASKS.md
1A.6) over the real 60 ExtraSensory subjects and commit the result. Contains
only UUIDs (already public dataset identifiers) and no sensor data, so
`splits/*.json` is tracked normally, unlike `data/`.

Usage: python scripts/make_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `python scripts/make_splits.py` finds the scripts/ package

from scripts.fetch_data import DEFAULT_OUT, META_DIR_NAME, list_subjects  # noqa: E402

from ats.splits import make_splits, save_splits  # noqa: E402
OUT_PATH = REPO_ROOT / "splits" / "subject_splits.json"
SEED = 0  # frozen alongside the ratios; changing it re-splits every subject


def main() -> None:
    meta_dir = DEFAULT_OUT / META_DIR_NAME
    subjects = list_subjects(meta_dir)
    splits = make_splits(subjects, seed=SEED)
    save_splits(splits, OUT_PATH)
    print(
        f"wrote {OUT_PATH.relative_to(REPO_ROOT)}: "
        f"{len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test "
        f"(of {len(subjects)} subjects)"
    )


if __name__ == "__main__":
    main()
