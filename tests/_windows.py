"""Builders for hand-made window tracks shared by the reasoning tests."""

from ats.contracts import CANONICAL_CLASSES, validate_window_track

# Per-activity signal levels. The resting postures sit under the stillness
# thresholds in ats/signal.py, as real sitting and lying do.
ACC_STD = {
    "LYING": 0.010,
    "SITTING": 0.012,
    "STANDING_STILL": 0.015,
    "STANDING_MOVING": 0.45,
    "WALKING": 1.55,
    "RUNNING": 4.20,
    "BICYCLING": 0.95,
}
GYRO_PER_AXIS = {
    "LYING": 0.0005,
    "SITTING": 0.0008,
    "STANDING_STILL": 0.001,
    "STANDING_MOVING": 0.2,
    "WALKING": 0.6,
    "RUNNING": 1.4,
    "BICYCLING": 0.8,
}
CADENCE = {"STANDING_MOVING": 0.4, "WALKING": 1.9, "RUNNING": 2.8, "BICYCLING": 1.2}


def make_window(index, t_start, activity, *, length=4.0, coverage=1.0, confidence=0.9, signal=None):
    """`signal` gives the window another activity's signal levels, e.g. a
    window labelled bicycling whose sensors are as still as sitting."""
    signal = signal or activity
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
            "acc_mag_std": ACC_STD[signal],
            "dominant_cadence_hz": CADENCE.get(signal, 0.0),
            "gyro_energy_x": GYRO_PER_AXIS[signal],
            "gyro_energy_y": GYRO_PER_AXIS[signal],
            "gyro_energy_z": GYRO_PER_AXIS[signal],
        },
        "model_id": "test",
    }
    validate_window_track(entry)
    return entry


def burst(minute_start, activity, index0=0, signal=None):
    """One ExtraSensory-style burst: ten 4 s windows at a 2 s hop, [m, m+22)."""
    return [make_window(index0 + i, minute_start + 2.0 * i, activity, signal=signal) for i in range(10)]


def minutes(plan):
    """(minute_start, activity[, signal]) tuples -> the concatenated bursts."""
    windows = []
    for minute_start, activity, *signal in plan:
        windows.extend(burst(minute_start, activity, index0=len(windows), signal=signal[0] if signal else None))
    return windows


def contiguous(activities, *, length=10.0, start=0.0, coverage=1.0):
    """Back-to-back, non-overlapping windows, one per label."""
    return [
        make_window(i, start + i * length, act, length=length, coverage=coverage)
        for i, act in enumerate(activities)
    ]
