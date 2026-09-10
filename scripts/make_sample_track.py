"""Regenerate the committed real-subject sample track (docs/TASKS.md 1A.8 /
"Artifacts crossing the boundary": "A -> B: ats/oracle.py + one committed
sample track"). Not the whole subject -- a full track is ~20k windows and
several MB -- but an excerpt covering every activity actually present in the
subject's recording, so Member B has real (not just synthetic) multi-class
data to exercise the reasoning/eval stack against ahead of Phase 3.

Requires the ExtraSensory archives staged first: `python scripts/fetch_data.py`.

Usage: python scripts/make_sample_track.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ats.contracts import CANONICAL_CLASSES
from ats.oracle import build_track

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBJECT = "00EABED2-271D-49D8-B599-1D4A09240601"
OUT_PATH = REPO_ROOT / "tests" / "fixtures" / f"track_subj_real_{SUBJECT[:8]}.jsonl"
WINDOWS_PER_ACTIVITY = 15


def main() -> None:
    entries = build_track(SUBJECT, REPO_ROOT / "data" / "raw")

    by_label: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        label = CANONICAL_CLASSES[entry["probs"].index(max(entry["probs"]))]
        by_label[label].append(entry)

    sample = [e for rows in by_label.values() for e in rows[:WINDOWS_PER_ACTIVITY]]
    sample.sort(key=lambda e: e["t_start"])

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for entry in sample:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    print(f"wrote {len(sample)} windows -> {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"activities present: {sorted(by_label)}")


if __name__ == "__main__":
    main()
