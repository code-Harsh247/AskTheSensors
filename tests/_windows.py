"""Builders for hand-made window tracks shared by the reasoning tests."""

from ats.contracts import CANONICAL_CLASSES, validate_window_track

ACC_STD = {
    "LYING": 0.03,
    "SITTING": 0.08,
    "STANDING_STILL": 0.12,
    "STANDING_MOVING": 0.45,
    "WALKING": 1.55,
    "RUNNING": 4.20,
    "BICYCLING": 0.95,
}
CADENCE = {"STANDING_MOVING": 0.4, "WALKING": 1.9, "RUNNING": 2.8, "BICYCLING": 1.2}


def make_window(index, t_start, activity, *, length=4.0, coverage=1.0, confidence=0.9):
    remainder = (1.0 - confidence) / (len(CANONICAL_CLASSES) - 1)
    probs = [remainder] * len(CANONICAL_CLASSES)
    probs[CANONICAL_CLASSES.index(activity)] = confidence
    entry = {
        "window_id": f"w{index:05d}",
        "t_start": t_start,
        "t_end": t_start + length,
        "probs": probs,
        "coverage": coverage,
        "feature_summary": {
            "acc_mag_mean": 9.8,
            "acc_mag_std": ACC_STD[activity],
            "dominant_cadence_hz": CADENCE.get(activity, 0.0),
            "gyro_energy_x": ACC_STD[activity],
            "gyro_energy_y": ACC_STD[activity],
            "gyro_energy_z": ACC_STD[activity],
        },
        "model_id": "test",
    }
    validate_window_track(entry)
    return entry


def burst(minute_start, activity, index0=0):
    """One ExtraSensory-style burst: ten 4 s windows at a 2 s hop, [m, m+22)."""
    return [make_window(index0 + i, minute_start + 2.0 * i, activity) for i in range(10)]


def minutes(plan):
    """(minute_start, activity) pairs -> the concatenated bursts."""
    windows = []
    for minute_start, activity in plan:
        windows.extend(burst(minute_start, activity, index0=len(windows)))
    return windows


def contiguous(activities, *, length=10.0, start=0.0, coverage=1.0):
    """Back-to-back, non-overlapping windows, one per label."""
    return [
        make_window(i, start + i * length, act, length=length, coverage=coverage)
        for i, act in enumerate(activities)
    ]
