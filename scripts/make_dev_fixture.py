"""Generate synthetic window tracks and the dev question set from one declared
ground truth.

Everything downstream (tracks and gold answers alike) is derived from
GROUND_TRUTH below, and never from ats.aggregate, so using these questions to
test the aggregation layer is not circular.

These subjects are SYNTHETIC fixtures, not ExtraSensory recordings. They exist
so Member B's stack can be exercised end to end before Member A's data
pipeline and oracle land. Real-subject questions get added in Phase 3.

Usage:  python scripts/make_dev_fixture.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
QUESTIONS_PATH = REPO_ROOT / "data" / "questions_dev.json"

WINDOW_SECONDS = 10.0
GAP = "__GAP__"

# (activity, t_start, t_end). GAP segments emit no windows at all, standing in
# for a stretch of missing recording.
GROUND_TRUTH: dict[str, list[tuple[str, float, float]]] = {
    # Walking split across a gap; no bicycling and no standing at all.
    "subj_synth_a": [
        ("SITTING", 0.0, 200.0),
        ("WALKING", 200.0, 320.0),
        (GAP, 320.0, 360.0),
        ("WALKING", 360.0, 420.0),
        ("RUNNING", 420.0, 500.0),
        ("LYING", 500.0, 600.0),
    ],
    # Walking and running totals are exactly equal: a genuine comparison tie.
    "subj_synth_b": [
        ("STANDING_STILL", 0.0, 100.0),
        ("WALKING", 100.0, 220.0),
        ("SITTING", 220.0, 300.0),
        ("RUNNING", 300.0, 420.0),
        ("STANDING_MOVING", 420.0, 480.0),
        ("LYING", 480.0, 600.0),
    ],
    # Prolonged lying, plus bicycling split across a gap.
    "subj_synth_c": [
        ("LYING", 0.0, 400.0),
        ("SITTING", 400.0, 450.0),
        ("BICYCLING", 450.0, 600.0),
        (GAP, 600.0, 660.0),
        ("BICYCLING", 660.0, 720.0),
        ("WALKING", 720.0, 800.0),
    ],
}

CANONICAL_CLASSES = (
    "LYING",
    "SITTING",
    "STANDING_STILL",
    "STANDING_MOVING",
    "WALKING",
    "RUNNING",
    "BICYCLING",
)

# Plausible per-activity signal characteristics for the synthetic fixture.
FEATURE_PROFILE = {
    "LYING": (9.79, 0.03, 0.0, 0.01),
    "SITTING": (9.80, 0.08, 0.0, 0.02),
    "STANDING_STILL": (9.81, 0.12, 0.0, 0.04),
    "STANDING_MOVING": (9.85, 0.45, 0.4, 0.20),
    "WALKING": (10.4, 1.55, 1.9, 0.55),
    "RUNNING": (12.6, 4.20, 2.8, 1.40),
    "BICYCLING": (10.0, 0.95, 1.2, 0.80),
}


def pretty(activity: str) -> str:
    return activity.replace("_", " ").title()


def _windows_for(subject: str, rng: random.Random) -> list[dict]:
    windows: list[dict] = []
    index = 0
    for activity, start, end in GROUND_TRUTH[subject]:
        if activity == GAP:
            continue
        t = start
        while t < end:
            mag_mean, mag_std, cadence, gyro = FEATURE_PROFILE[activity]
            true_index = CANONICAL_CLASSES.index(activity)
            confidence = 0.80 + rng.uniform(0.0, 0.10)
            remainder = (1.0 - confidence) / (len(CANONICAL_CLASSES) - 1)
            probs = [remainder] * len(CANONICAL_CLASSES)
            probs[true_index] = confidence
            windows.append(
                {
                    "window_id": f"{subject}_w{index:04d}",
                    "t_start": round(t, 3),
                    "t_end": round(min(t + WINDOW_SECONDS, end), 3),
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
            )
            t += WINDOW_SECONDS
            index += 1
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


def _evidence_gold(intervals: list[list[float]]) -> dict:
    return {
        "cited_intervals": intervals,
        "modality": "both",
        "channels": ["all"],
    }


def _q(qid: str, text: str, gold: dict) -> dict:
    return {"question_id": qid, "text": text, "gold": gold}


def build_questions() -> list[dict]:
    questions: list[dict] = []

    for subject in GROUND_TRUTH:
        present = [act for act, _, _ in segments_of(subject)]
        seen: list[str] = []
        for act in present:
            if act not in seen:
                seen.append(act)
        absent = [c for c in CANONICAL_CLASSES if c not in seen]

        # --- Tier 1: identification and verification -----------------------
        probe_times = [
            segments_of(subject)[1][1] + 20.0,
            segments_of(subject)[-1][1] + 20.0,
        ]
        for n, t in enumerate(probe_times):
            act = activity_at(subject, t)
            questions.append(
                _q(
                    f"{subject}_t1_id{n}",
                    f"What activity is the user performing at {t:g} seconds?",
                    {
                        "answer": pretty(act),
                        "activity_event": pretty(act),
                        "question_type": "identification",
                        "answer_kind": "categorical",
                    },
                )
            )

        verify_target = seen[1]
        t_hit = [s for a, s, _ in segments_of(subject) if a == verify_target][0] + 10.0
        questions.append(
            _q(
                f"{subject}_t1_ver0",
                f"Is the user {pretty(verify_target).lower()} at {t_hit:g} seconds?",
                {
                    "answer": "Yes",
                    "activity_event": pretty(verify_target),
                    "question_type": "verification",
                    "answer_kind": "categorical",
                },
            )
        )
        miss_target = absent[0] if absent else seen[-1]
        questions.append(
            _q(
                f"{subject}_t1_ver1",
                f"Is the user {pretty(miss_target).lower()} at {t_hit:g} seconds?",
                {
                    "answer": "No",
                    "activity_event": pretty(miss_target),
                    "question_type": "verification",
                    "answer_kind": "categorical",
                },
            )
        )

        # --- Tier 2: duration, count, comparison ---------------------------
        for n, act in enumerate(seen[:2]):
            total = total_duration(subject, act)
            questions.append(
                _q(
                    f"{subject}_t2_dur{n}",
                    f"How long was the user {pretty(act).lower()} in total?",
                    {
                        "answer": f"{total:g} seconds",
                        "activity_event": pretty(act),
                        "question_type": "duration",
                        "answer_kind": "numeric",
                        "numeric_value": total,
                        **_evidence_gold(intervals_of(subject, act)),
                    },
                )
            )

        # Edge case: an activity that never occurs in this recording.
        if absent:
            questions.append(
                _q(
                    f"{subject}_t2_dur_absent",
                    f"How long was the user {pretty(absent[0]).lower()} in total?",
                    {
                        "answer": "0 seconds",
                        "activity_event": pretty(absent[0]),
                        "question_type": "duration",
                        "answer_kind": "numeric",
                        "numeric_value": 0.0,
                    },
                )
            )

        split_act = next(
            (a for a in seen if len(intervals_of(subject, a)) > 1), seen[0]
        )
        questions.append(
            _q(
                f"{subject}_t2_count0",
                f"How many separate times did the user do {pretty(split_act).lower()}?",
                {
                    "answer": str(len(intervals_of(subject, split_act))),
                    "activity_event": pretty(split_act),
                    "question_type": "count",
                    "answer_kind": "numeric",
                    "numeric_value": float(len(intervals_of(subject, split_act))),
                    **_evidence_gold(intervals_of(subject, split_act)),
                },
            )
        )

        a_act, b_act = seen[0], seen[1]
        a_total, b_total = total_duration(subject, a_act), total_duration(subject, b_act)
        if a_total == b_total:
            verdict = "Equal"
        else:
            verdict = pretty(a_act if a_total > b_total else b_act)
        questions.append(
            _q(
                f"{subject}_t2_cmp0",
                f"Did the user spend more time {pretty(a_act).lower()} or {pretty(b_act).lower()}?",
                {
                    "answer": verdict,
                    "activity_event": f"{pretty(a_act)}, {pretty(b_act)}",
                    "question_type": "comparison",
                    "answer_kind": "categorical",
                },
            )
        )

        # Edge case: exact tie between two activity totals.
        for x in seen:
            for y in seen:
                if x < y and total_duration(subject, x) == total_duration(subject, y):
                    questions.append(
                        _q(
                            f"{subject}_t2_cmp_tie",
                            f"Did the user spend more time {pretty(x).lower()} or {pretty(y).lower()}?",
                            {
                                "answer": "Equal",
                                "activity_event": f"{pretty(x)}, {pretty(y)}",
                                "question_type": "comparison",
                                "answer_kind": "categorical",
                            },
                        )
                    )
                    break
            else:
                continue
            break

        # --- Tier 3: evidence grounding ------------------------------------
        for n, act in enumerate(seen[:3]):
            ivs = intervals_of(subject, act)
            questions.append(
                _q(
                    f"{subject}_t3_ground{n}",
                    f"Cite the stretch of signal where the user was {pretty(act).lower()}.",
                    {
                        "answer": pretty(act),
                        "activity_event": pretty(act),
                        "question_type": "grounding",
                        "answer_kind": "temporal",
                        **_evidence_gold(ivs),
                    },
                )
            )

        onset_act = seen[1]
        onset = intervals_of(subject, onset_act)[0]
        questions.append(
            _q(
                f"{subject}_t3_onset",
                f"Did the user begin {pretty(onset_act).lower()} at any point, and if so, when?",
                {
                    "answer": f"Yes, {pretty(onset_act).lower()} began at {onset[0]:g} seconds",
                    "activity_event": f"Onset of {pretty(onset_act).lower()}",
                    "question_type": "grounding",
                    "answer_kind": "temporal",
                    **_evidence_gold([onset]),
                },
            )
        )

        # --- Tier 4: open-world reasoning ----------------------------------
        lying_total = total_duration(subject, "LYING")
        questions.append(
            _q(
                f"{subject}_t4_rest",
                "Did the user lie down for a prolonged period?",
                {
                    "answer": "Likely yes" if lying_total >= 300 else "Likely no",
                    "activity_event": "Prolonged lying down",
                    "question_type": "open_world",
                    "answer_kind": "open_world",
                    **_evidence_gold(intervals_of(subject, "LYING")),
                },
            )
        )

        cycling_total = total_duration(subject, "BICYCLING")
        questions.append(
            _q(
                f"{subject}_t4_wheeled",
                "Was the user using a wheeled or pedal-based mode of movement?",
                {
                    "answer": "Yes" if cycling_total > 0 else "No",
                    "activity_event": "Unknown outdoor physical activity, consistent with cycling"
                    if cycling_total > 0
                    else "No wheeled movement observed",
                    "question_type": "open_world",
                    "answer_kind": "open_world",
                    **_evidence_gold(intervals_of(subject, "BICYCLING")),
                },
            )
        )

        active = sum(
            total_duration(subject, a) for a in ("WALKING", "RUNNING", "BICYCLING")
        )
        sedentary = sum(
            total_duration(subject, a) for a in ("LYING", "SITTING", "STANDING_STILL")
        )
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

        # Edge case: a question whose answer lands inside a data gap.
        gaps = gaps_of(subject)
        if gaps:
            gap_start, gap_end = gaps[0]
            midpoint = (gap_start + gap_end) / 2
            questions.append(
                _q(
                    f"{subject}_t4_gap",
                    f"What was the user doing at {midpoint:g} seconds?",
                    {
                        "answer": "N/A",
                        "activity_event": "No data",
                        "question_type": "open_world",
                        "answer_kind": "open_world",
                    },
                )
            )
        else:
            longest = max(segments_of(subject), key=lambda s: s[2] - s[1])
            questions.append(
                _q(
                    f"{subject}_t4_longest",
                    "Which single stretch of the recording shows the least movement, and why?",
                    {
                        "answer": pretty(longest[0]),
                        "activity_event": "Lowest-movement stretch",
                        "question_type": "open_world",
                        "answer_kind": "open_world",
                        **_evidence_gold([[longest[1], longest[2]]]),
                    },
                )
            )

    return questions


def main() -> None:
    rng = random.Random(20260909)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for subject in GROUND_TRUTH:
        path = FIXTURE_DIR / f"track_{subject}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for window in _windows_for(subject, rng):
                f.write(json.dumps(window, sort_keys=True) + "\n")
        print(f"wrote {path}")

    questions = build_questions()
    payload = {
        "description": (
            "Dev question set over SYNTHETIC fixture subjects (subj_synth_a/b/c), "
            "generated by scripts/make_dev_fixture.py. Gold is derived from the "
            "declared ground truth in that script, not from ats.aggregate. These "
            "are not ExtraSensory recordings; real-subject questions are added "
            "once Member A's Phase 1 data pipeline lands."
        ),
        "questions": questions,
    }
    with QUESTIONS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {QUESTIONS_PATH} ({len(questions)} questions)")


if __name__ == "__main__":
    main()
