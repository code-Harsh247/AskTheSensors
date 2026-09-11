"""docs/bug.md issue 1, verification checklist item 1: confirm every subject's
converted accelerometer magnitude lands near gravity, and record the
per-subject units decision `ats.ingest._detect_acc_scale` made along the way.

Reads only the raw_acc archive (not gyro or labels) since the units decision
depends on accelerometer data alone -- much faster than a full `load_subject`
pass over all 60 subjects.

Usage: python scripts/check_acc_units.py [--data-dir data/raw]
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ats.ingest import (  # noqa: E402
    _CONVERTED_SANITY_RANGE,
    _G_LIKE_RANGE,
    _MS2_LIKE_RANGE,
    _iter_bursts,
    _read_dat_bytes,
    _sanitize_burst_rows,
    G_TO_MS2,
)
from scripts.fetch_data import DEFAULT_OUT, META_DIR_NAME, list_subjects  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "results" / "acc_units_by_subject.csv"


def _decision_for(raw_median: float) -> tuple[str, float]:
    """Mirrors ats.ingest._detect_acc_scale's classification, but also names
    the "ambiguous" case explicitly instead of silently defaulting to 1.0."""
    if _G_LIKE_RANGE[0] <= raw_median <= _G_LIKE_RANGE[1]:
        return "g", G_TO_MS2
    if _MS2_LIKE_RANGE[0] <= raw_median <= _MS2_LIKE_RANGE[1]:
        return "ms2", 1.0
    return "ambiguous (left unconverted)", 1.0


def analyze_subject(meta_dir: Path, subject_id: str) -> dict | None:
    raw_rows = [
        _sanitize_burst_rows(_read_dat_bytes(data), subject_id, ts)
        for ts, data in _iter_bursts(meta_dir, "raw_acc", subject_id, ".m_raw_acc.dat")
    ]
    magnitudes = [math.sqrt(x * x + y * y + z * z) for rows in raw_rows for _, x, y, z in rows]
    if not magnitudes:
        return None
    raw_median = statistics.median(magnitudes)
    decision, scale = _decision_for(raw_median)
    converted_median = raw_median * scale
    within_range = _CONVERTED_SANITY_RANGE[0] <= converted_median <= _CONVERTED_SANITY_RANGE[1]
    return {
        "subject_id": subject_id,
        "raw_median_magnitude": round(raw_median, 4),
        "units_decision": decision,
        "converted_median_magnitude": round(converted_median, 4),
        "within_gravity_range": within_range,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    meta_dir = Path(args.data_dir) / META_DIR_NAME
    subjects = list_subjects(meta_dir)

    rows = []
    for subject_id in subjects:
        row = analyze_subject(meta_dir, subject_id)
        if row is None:
            print(f"WARNING: subject {subject_id} has no accelerometer data at all -- skipping")
            continue
        rows.append(row)
        flag = "" if row["within_gravity_range"] else "  <-- OUT OF RANGE"
        print(
            f"{subject_id}: raw median {row['raw_median_magnitude']:.3f}, "
            f"decision={row['units_decision']}, "
            f"converted median {row['converted_median_magnitude']:.3f} m/s^2{flag}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_of_range = [r for r in rows if not r["within_gravity_range"]]
    print(f"\nwrote {len(rows)} subjects -> {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"{len(out_of_range)} subject(s) outside the 8-12 m/s^2 gravity range after conversion.")


if __name__ == "__main__":
    main()
