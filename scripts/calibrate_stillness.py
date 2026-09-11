"""Calibrate the stillness predicate on subj_real_a and check it on
subj_real_b (docs/TASKS.md task 4B.3).

The thresholds come only from subj_real_a's lying-down windows. subj_real_b is
never used to choose them; it shows how the rule holds on a subject it was not
fitted to, including whether real running and bicycling survive the
moving-signal requirement. Writes results/stillness_calibration.json.

Usage:  python scripts/calibrate_stillness.py
"""

from __future__ import annotations

import json
from collections import defaultdict

from ats.aggregate import build_timeline, load_track
from ats.contracts import CANONICAL_CLASSES
from ats.eval.dev import FIXTURES_DIR, REPO_ROOT
from ats.evidence import summarize_each
from ats.signal import (
    GRAVITY_BAND,
    MAJORITY,
    STILL_ACC_STD,
    STILL_GYRO_ENERGY,
    calibrate_still_thresholds,
    is_still,
    recording_gravity,
)

CALIBRATION = "subj_real_a"
CHECK = "subj_real_b"


def _label(window: dict) -> str:
    probs = window["probs"]
    return CANONICAL_CLASSES[max(range(len(probs)), key=probs.__getitem__)]


def window_still_share(windows: list[dict]) -> dict[str, dict]:
    by_class: dict[str, list[bool]] = defaultdict(list)
    for w in windows:
        if w["coverage"] >= 0.95:
            by_class[_label(w)].append(is_still(w))
    return {c: {"n": len(v), "still_share": sum(v) / len(v)} for c, v in by_class.items()}


def interval_still_share(windows: list[dict]) -> dict[str, dict]:
    """Per class, the share of timeline intervals whose windows are mostly
    still -- the unit the operators actually decide on."""
    timeline = build_timeline(windows)
    summaries = summarize_each(windows, [iv.as_tuple() for iv in timeline.intervals])
    by_class: dict[str, list[bool]] = defaultdict(list)
    for interval, summary in zip(timeline.intervals, summaries):
        if summary is not None:
            by_class[interval.activity].append(summary.still_share >= MAJORITY)
    return {c: {"intervals": len(v), "mostly_still": sum(v) / len(v)} for c, v in by_class.items()}


def main() -> None:
    calibration = load_track(FIXTURES_DIR / f"track_{CALIBRATION}.jsonl")
    derived = calibrate_still_thresholds(calibration)
    if derived != (STILL_ACC_STD, STILL_GYRO_ENERGY):
        raise SystemExit(f"ats/signal.py thresholds {(STILL_ACC_STD, STILL_GYRO_ENERGY)} differ from calibration {derived}")

    check = load_track(FIXTURES_DIR / f"track_{CHECK}.jsonl")
    report = {
        "thresholds": {"acc_mag_std": STILL_ACC_STD, "gyro_energy": STILL_GYRO_ENERGY, "calibrated_on": CALIBRATION},
        "median_acc_magnitude_m_s2": {
            CALIBRATION: recording_gravity(calibration),
            CHECK: recording_gravity(check),
            "plausible_band": list(GRAVITY_BAND),
        },
        "windows_judged_still_by_true_class": {
            CALIBRATION: window_still_share(calibration),
            CHECK: window_still_share(check),
        },
        "intervals_judged_mostly_still_by_true_class": {
            CALIBRATION: interval_still_share(calibration),
            CHECK: interval_still_share(check),
        },
    }
    out = REPO_ROOT / "results" / "stillness_calibration.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"thresholds: acc_mag_std <= {STILL_ACC_STD}, gyro_energy <= {STILL_GYRO_ENERGY} (calibrated on {CALIBRATION})")
    for subject in (CALIBRATION, CHECK):
        gravity = report["median_acc_magnitude_m_s2"][subject]
        plausible = GRAVITY_BAND[0] <= gravity <= GRAVITY_BAND[1]
        print(f"{subject}: median acceleration magnitude {gravity:.2f} m/s^2 ({'near gravity' if plausible else 'NOT near gravity: units problem'})")
    for subject in (CALIBRATION, CHECK):
        role = "calibration" if subject == CALIBRATION else "check, not used to calibrate"
        windows = report["windows_judged_still_by_true_class"][subject]
        intervals = report["intervals_judged_mostly_still_by_true_class"][subject]
        print(f"--- {subject} ({role}): share judged still, per true class")
        for c in CANONICAL_CLASSES:
            if c in windows:
                iv = intervals.get(c, {"mostly_still": float("nan"), "intervals": 0})
                print(
                    f"   {c:16s} windows {windows[c]['still_share']:.2f} (n={windows[c]['n']})   "
                    f"intervals {iv['mostly_still']:.2f} (n={iv['intervals']})"
                )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
