"""Generate synthetic window tracks and per-subject dev question sets from one
declared ground truth.

Tracks and gold answers are both derived from GROUND_TRUTH below, never from
ats.aggregate or ats.operators, so these questions test the reasoning stack
without circularity. What is shared with the system is vocabulary and
definitions only (display names, what counts as prolonged, which activities
are at rest), imported from ats.vocab.

The tracks mimic ExtraSensory's real recording structure (compare
tests/fixtures/track_subj_real_00EABED2.jsonl): one ~22 s burst of 4 s /
2 s-hop windows at the start of every labeled minute. A missing-data gap is a
run of whole minutes with no burst at all. Under the team's minute-attribution
decision (docs/TASKS.md §0), each segment's gold interval is exactly its
declared minutes.

These subjects are SYNTHETIC fixtures, not ExtraSensory recordings; questions
over real subjects are added in Phase 3.

Usage:  python scripts/make_dev_fixture.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ats.contracts import CANONICAL_CLASSES
from ats.vocab import ACTIVE, DISPLAY_NAMES, PROLONGED_S, SEDENTARY

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
QUESTIONS_DIR = REPO_ROOT / "data" / "questions_dev"

MINUTE_S = 60.0
BURST_S = 22.0
WINDOW_S = 4.0
HOP_S = 2.0
GAP = "__GAP__"

# (activity, t_start, t_end) in seconds, always whole minutes.
GROUND_TRUTH: dict[str, list[tuple[str, float, float]]] = {
    # Walking split across a four-minute gap; no bicycling or standing at all.
    "subj_synth_a": [
        ("SITTING", 0.0, 600.0),
        ("WALKING", 600.0, 1080.0),
        (GAP, 1080.0, 1320.0),
        ("WALKING", 1320.0, 1560.0),
        ("RUNNING", 1560.0, 1860.0),
        ("LYING", 1860.0, 2100.0),
    ],
    # Walking and running totals are exactly equal: a genuine comparison tie.
    "subj_synth_b": [
        ("STANDING_STILL", 0.0, 300.0),
        ("WALKING", 300.0, 720.0),
        ("SITTING", 720.0, 960.0),
        ("RUNNING", 960.0, 1380.0),
        ("STANDING_MOVING", 1380.0, 1560.0),
        ("LYING", 1560.0, 1800.0),
    ],
    # Twenty minutes of lying, then bicycling split across a three-minute gap.
    "subj_synth_c": [
        ("LYING", 0.0, 1200.0),
        ("SITTING", 1200.0, 1500.0),
        ("BICYCLING", 1500.0, 2100.0),
        (GAP, 2100.0, 2280.0),
        ("BICYCLING", 2280.0, 2640.0),
        ("WALKING", 2640.0, 2940.0),
    ],
}

# Plausible per-activity signal characteristics:
# (acc magnitude mean, acc magnitude std, dominant cadence Hz, gyro energy).
FEATURE_PROFILE = {
    "LYING": (9.79, 0.03, 0.0, 0.01),
    "SITTING": (9.80, 0.08, 0.0, 0.02),
    "STANDING_STILL": (9.81, 0.12, 0.0, 0.04),
    "STANDING_MOVING": (9.85, 0.45, 0.4, 0.20),
    "WALKING": (10.4, 1.55, 1.9, 0.55),
    "RUNNING": (12.6, 4.20, 2.8, 1.40),
    "BICYCLING": (10.0, 0.95, 1.2, 0.80),
}


def name(activity: str) -> str:
    return DISPLAY_NAMES[activity]


def phrase(activity: str) -> str:
    return DISPLAY_NAMES[activity].lower()


def _window(subject: str, index: int, t: float, activity: str, rng: random.Random) -> dict:
    mag_mean, mag_std, cadence, gyro = FEATURE_PROFILE[activity]
    confidence = 0.80 + rng.uniform(0.0, 0.10)
    remainder = (1.0 - confidence) / (len(CANONICAL_CLASSES) - 1)
    probs = [remainder] * len(CANONICAL_CLASSES)
    probs[CANONICAL_CLASSES.index(activity)] = confidence
    return {
        "window_id": f"{subject}_w{index:05d}",
        "t_start": round(t, 3),
        "t_end": round(t + WINDOW_S, 3),
        "probs": [round(p, 6) for p in probs],
        "coverage": 1.0,
        "feature_summary": {
            "acc_mag_mean": round(mag_mean + rng.uniform(-0.05, 0.05), 4),
            "acc_mag_std": round(mag_std + rng.uniform(-0.02, 0.02), 4),
            "dominant_cadence_hz": round(cadence, 4),
            "gyro_energy_x": round(gyro + rng.uniform(0, 0.05), 4),
            "gyro_energy_y": round(gyro + rng.uniform(0, 0.05), 4),
            "gyro_energy_z": round(gyro * 0.5 + rng.uniform(0, 0.05), 4),
        },
        "model_id": "synthetic-fixture",
    }


def _windows_for(subject: str, rng: random.Random) -> list[dict]:
    windows: list[dict] = []
    for activity, start, end in GROUND_TRUTH[subject]:
        if activity == GAP:
            continue
        minute = start
        while minute < end - 1e-9:
            t = minute
            while t + WINDOW_S <= minute + BURST_S + 1e-9:
                windows.append(_window(subject, len(windows), t, activity, rng))
                t += HOP_S
            minute += MINUTE_S
    return windows


def segments_of(subject: str) -> list[tuple[str, float, float]]:
    return [s for s in GROUND_TRUTH[subject] if s[0] != GAP]


def intervals_of(subject: str, activity: str) -> list[list[float]]:
    return [[s, e] for act, s, e in segments_of(subject) if act == activity]


def total_duration(subject: str, activity: str) -> float:
    return sum(e - s for act, s, e in segments_of(subject) if act == activity)


def activity_at(subject: str, t: float) -> str | None:
    for act, s, e in segments_of(subject):
        if s <= t < e:
            return act
    return None


def gaps_of(subject: str) -> list[list[float]]:
    return [[s, e] for act, s, e in GROUND_TRUTH[subject] if act == GAP]


def _evidence(intervals: list[list[float]]) -> dict:
    return {"cited_intervals": intervals, "modality": "both", "channels": ["all"]}


def _q(qid: str, text: str, gold: dict) -> dict:
    return {"question_id": qid, "text": text, "gold": gold}


def build_questions(subject: str) -> list[dict]:
    questions: list[dict] = []
    segments = segments_of(subject)
    seen: list[str] = []
    for act, _, _ in segments:
        if act not in seen:
            seen.append(act)
    absent = [c for c in CANONICAL_CLASSES if c not in seen]

    # --- Tier 1: identification and verification ---------------------------
    for n, t in enumerate([segments[1][1] + 20.0, segments[-1][1] + 20.0]):
        act = activity_at(subject, t)
        questions.append(
            _q(
                f"{subject}_t1_id{n}",
                f"What activity is the user performing at {t:g} seconds?",
                {"answer": name(act), "activity_event": name(act), "question_type": "identification", "answer_kind": "categorical"},
            )
        )

    target = seen[1]
    t_hit = intervals_of(subject, target)[0][0] + 10.0
    miss = next(c for c in CANONICAL_CLASSES if c != activity_at(subject, t_hit))
    for suffix, act, verdict in (("ver0", target, "Yes"), ("ver1", miss, "No")):
        questions.append(
            _q(
                f"{subject}_t1_{suffix}",
                f"Is the user {phrase(act)} at {t_hit:g} seconds?",
                {"answer": verdict, "activity_event": name(act), "question_type": "verification", "answer_kind": "categorical"},
            )
        )

    # --- Tier 2: duration, count, comparison -------------------------------
    for n, act in enumerate(seen[:2]):
        total = total_duration(subject, act)
        questions.append(
            _q(
                f"{subject}_t2_dur{n}",
                f"How long was the user {phrase(act)} in total?",
                {
                    "answer": f"{total:g} seconds",
                    "activity_event": name(act),
                    "question_type": "duration",
                    "answer_kind": "numeric",
                    "numeric_value": total,
                    **_evidence(intervals_of(subject, act)),
                },
            )
        )

    # Edge case: an activity that never occurs in this recording.
    if absent:
        questions.append(
            _q(
                f"{subject}_t2_dur_absent",
                f"How long was the user {phrase(absent[0])} in total?",
                {
                    "answer": "0 seconds",
                    "activity_event": name(absent[0]),
                    "question_type": "duration",
                    "answer_kind": "numeric",
                    "numeric_value": 0.0,
                },
            )
        )

    # Edge case (where present): an activity split across a data gap.
    split = next((a for a in seen if len(intervals_of(subject, a)) > 1), seen[0])
    bouts = intervals_of(subject, split)
    questions.append(
        _q(
            f"{subject}_t2_count0",
            f"How many separate times was the user {phrase(split)}?",
            {
                "answer": str(len(bouts)),
                "activity_event": name(split),
                "question_type": "count",
                "answer_kind": "numeric",
                "numeric_value": float(len(bouts)),
                **_evidence(bouts),
            },
        )
    )

    def comparison(qid: str, a: str, b: str) -> dict:
        total_a, total_b = total_duration(subject, a), total_duration(subject, b)
        verdict = "Equal" if total_a == total_b else name(a if total_a > total_b else b)
        return _q(
            qid,
            f"Did the user spend more time {phrase(a)} or {phrase(b)}?",
            {"answer": verdict, "activity_event": f"{name(a)}, {name(b)}", "question_type": "comparison", "answer_kind": "categorical"},
        )

    questions.append(comparison(f"{subject}_t2_cmp0", seen[0], seen[1]))

    # Edge case: two activities with exactly equal totals.
    tie = next(
        (
            (a, b)
            for i, a in enumerate(seen)
            for b in seen[i + 1 :]
            if total_duration(subject, a) == total_duration(subject, b)
        ),
        None,
    )
    if tie:
        questions.append(comparison(f"{subject}_t2_cmp_tie", *tie))

    # --- Tier 3: evidence grounding ----------------------------------------
    for n, act in enumerate(seen[:3]):
        questions.append(
            _q(
                f"{subject}_t3_ground{n}",
                f"Cite the stretch of signal where the user was {phrase(act)}.",
                {
                    "answer": name(act),
                    "activity_event": name(act),
                    "question_type": "grounding",
                    "answer_kind": "temporal",
                    **_evidence(intervals_of(subject, act)),
                },
            )
        )

    onset_act = seen[1]
    onset = intervals_of(subject, onset_act)[0]
    questions.append(
        _q(
            f"{subject}_t3_onset",
            f"Did the user begin {phrase(onset_act)} at any point, and if so, when?",
            {
                "answer": f"Yes, {phrase(onset_act)} began at {onset[0]:g} seconds",
                "activity_event": f"Onset of {phrase(onset_act)}",
                "question_type": "grounding",
                "answer_kind": "temporal",
                **_evidence([onset]),
            },
        )
    )

    # --- Tier 4: open-world reasoning --------------------------------------
    lying = intervals_of(subject, "LYING")
    prolonged = [iv for iv in lying if iv[1] - iv[0] >= PROLONGED_S]
    questions.append(
        _q(
            f"{subject}_t4_rest",
            "Did the user lie down for a prolonged period?",
            {
                "answer": "Likely yes" if prolonged else "Likely no",
                "activity_event": "Prolonged lying down",
                "question_type": "open_world",
                "answer_kind": "open_world",
                **_evidence(prolonged or lying),
            },
        )
    )

    cycling = intervals_of(subject, "BICYCLING")
    questions.append(
        _q(
            f"{subject}_t4_wheeled",
            "Was the user using a wheeled or pedal-based mode of movement?",
            {
                "answer": "Yes" if cycling else "No",
                "activity_event": "Consistent with cycling" if cycling else "No wheeled movement observed",
                "question_type": "open_world",
                "answer_kind": "open_world",
                **_evidence(cycling),
            },
        )
    )

    sedentary = sum(total_duration(subject, a) for a in SEDENTARY)
    active = sum(total_duration(subject, a) for a in ACTIVE)
    questions.append(
        _q(
            f"{subject}_t4_balance",
            "Was the user mostly at rest or mostly physically active during this recording?",
            {
                "answer": "Mostly at rest" if sedentary > active else "Mostly active",
                "activity_event": "Overall activity balance",
                "question_type": "open_world",
                "answer_kind": "open_world",
            },
        )
    )

    gaps = gaps_of(subject)
    if gaps:
        # Edge case: the answer lands inside a data gap.
        midpoint = (gaps[0][0] + gaps[0][1]) / 2
        questions.append(
            _q(
                f"{subject}_t4_gap",
                f"What was the user doing at {midpoint:g} seconds?",
                {"answer": "N/A", "activity_event": "No data", "question_type": "open_world", "answer_kind": "open_world"},
            )
        )
    else:
        stillest = min(segments, key=lambda s: (FEATURE_PROFILE[s[0]][1], -(s[2] - s[1])))
        questions.append(
            _q(
                f"{subject}_t4_least_movement",
                "Which single stretch of the recording shows the least movement, and why?",
                {
                    "answer": name(stillest[0]),
                    "activity_event": "Lowest-movement stretch",
                    "question_type": "open_world",
                    "answer_kind": "open_world",
                    **_evidence([[stillest[1], stillest[2]]]),
                },
            )
        )

    return questions


def main() -> None:
    rng = random.Random(20260909)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for subject in GROUND_TRUTH:
        track_path = FIXTURE_DIR / f"track_{subject}.jsonl"
        windows = _windows_for(subject, rng)
        with track_path.open("w", encoding="utf-8") as f:
            for window in windows:
                f.write(json.dumps(window, sort_keys=True) + "\n")

        questions = build_questions(subject)
        payload = {
            "description": (
                f"Dev questions for {subject}, a SYNTHETIC fixture subject generated by "
                "scripts/make_dev_fixture.py. Gold is derived from the declared ground truth "
                "in that script, not from the reasoning code. Not an ExtraSensory recording."
            ),
            "questions": questions,
        }
        question_path = QUESTIONS_DIR / f"{subject}.json"
        with question_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"{subject}: {len(windows)} windows -> {track_path.name}, {len(questions)} questions -> {question_path.name}")


if __name__ == "__main__":
    main()
