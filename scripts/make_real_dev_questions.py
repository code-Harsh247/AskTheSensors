"""Author the dev question sets for the two real ExtraSensory subjects handed
over for Phase 3 (docs/phase3_handoff.md): data/questions_dev_v2/subj_real_{a,b}.json.

Gold comes from each subject's oracle track, whose labels are the
ExtraSensory ground truth. The bouts are derived here by a second,
deliberately separate implementation of the minute-attribution rules in
docs/TASKS.md §0, and the script refuses to write anything unless it matches
ats.aggregate.build_timeline exactly, so the gold is checked against two
implementations instead of trusted from one.

Because gold and the oracle track share the same labels, the oracle is right
by construction: in the Phase 3 delta, every difference between the oracle
and the real-model track is a recognition difference.

Usage:  python scripts/make_real_dev_questions.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ats.aggregate import build_timeline, load_track
from ats.contracts import CANONICAL_CLASSES
from ats.serialize import format_seconds as fs
from ats.vocab import ACTIVE, DISPLAY_NAMES, PROLONGED_S, SEDENTARY

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
OUT_DIR = REPO_ROOT / "data" / "questions_dev_v2"
REAL_TRACK_DIR = FIXTURE_DIR / "real_model_tracks"

# Frozen in docs/TASKS.md §0.
CYCLE_S = 60.0
HALF_HOP_S = 1.0
MIN_COVERAGE = 0.5

UUIDS = {
    "subj_real_a": "00EABED2-271D-49D8-B599-1D4A09240601",
    "subj_real_b": "74B86067-5D4B-43CF-82CF-341B76BEA0F4",
}

# Chosen per subject so every question has an unambiguous answer and together
# they cover common, rare, and absent activities.
TARGETS = {
    "subj_real_a": {
        "identify": ("SITTING", "WALKING"),
        "verify": "LYING",
        "durations": ("SITTING", "STANDING_STILL"),
        "absent": "RUNNING",
        "count": "WALKING",
        "comparisons": (("SITTING", "LYING"),),
        "ground": ("STANDING_STILL", "WALKING", "STANDING_MOVING"),
        "onset": "WALKING",
    },
    "subj_real_b": {
        "identify": ("BICYCLING", "LYING"),
        "verify": "RUNNING",
        "durations": ("SITTING", "WALKING"),
        "absent": None,
        "count": "RUNNING",
        # Running exceeds walking for this subject, the opposite direction to
        # the brief's own worked example.
        "comparisons": (("WALKING", "RUNNING"), ("SITTING", "LYING")),
        "ground": ("RUNNING", "WALKING", "STANDING_STILL"),
        "onset": "BICYCLING",
    },
}

Bout = tuple[str, float, float]


def _label(group: list[dict]) -> str:
    totals = [sum(w["probs"][i] for w in group) for i in range(len(CANONICAL_CLASSES))]
    return CANONICAL_CLASSES[totals.index(max(totals))]


def truth_bouts(windows: list[dict]) -> tuple[list[Bout], list[tuple[float, float]]]:
    """Minute-attributed bouts straight from the labels (docs/TASKS.md §0)."""
    reliable = sorted((w for w in windows if w["coverage"] >= MIN_COVERAGE), key=lambda w: w["t_start"])
    bursts: list[list[dict]] = []
    for window in reliable:
        if bursts and window["t_start"] - bursts[-1][-1]["t_end"] <= HALF_HOP_S:
            bursts[-1].append(window)
        else:
            bursts.append([window])
    if any(b[-1]["t_end"] - b[0]["t_start"] > CYCLE_S for b in bursts):
        raise SystemExit("a burst spans more than one minute; minute attribution does not apply")

    pieces: list[list] = []
    gaps: list[tuple[float, float]] = []
    for i, burst in enumerate(bursts):
        start = burst[0]["t_start"]
        end = max(burst[-1]["t_end"], start + CYCLE_S)
        if i + 1 < len(bursts):
            following = bursts[i + 1][0]["t_start"]
            end = min(end, following)
            if following - end < CYCLE_S:
                end = following
            else:
                gaps.append((round(end, 3), round(following, 3)))
        pieces.append([_label(burst), start, end])

    bouts: list[list] = []
    for label, start, end in pieces:
        if bouts and bouts[-1][0] == label and bouts[-1][2] == start:
            bouts[-1][2] = end
        else:
            bouts.append([label, start, end])
    return [(a, round(s, 3), round(e, 3)) for a, s, e in bouts], gaps


def cross_check(subject: str, windows: list[dict], bouts: list[Bout], gaps: list[tuple[float, float]]) -> None:
    timeline = build_timeline(windows)
    produced = [(iv.activity, iv.t_start, iv.t_end) for iv in timeline.intervals]
    if produced != bouts or list(timeline.gaps) != gaps:
        first = next((i for i, (p, b) in enumerate(zip(produced, bouts)) if p != b), min(len(produced), len(bouts)))
        raise SystemExit(
            f"{subject}: ats.aggregate disagrees with the independent derivation at bout {first}: "
            f"aggregate {produced[first:first + 2]}, independent {bouts[first:first + 2]}"
        )


def name(activity: str) -> str:
    return DISPLAY_NAMES[activity]


def phrase(activity: str) -> str:
    return DISPLAY_NAMES[activity].lower()


def _evidence(intervals: list[list[float]]) -> dict:
    return {"cited_intervals": intervals, "modality": "both", "channels": ["all"]}


def _q(qid: str, text: str, gold: dict) -> dict:
    return {"question_id": qid, "text": text, "gold": gold}


def build_questions(
    subject: str,
    bouts: list[Bout],
    gaps: list[tuple[float, float]],
    recorded: list[tuple[float, float]],
) -> list[dict]:
    targets = TARGETS[subject]

    def of(activity: str) -> list[list[float]]:
        return [[s, e] for a, s, e in bouts if a == activity]

    def total(activity: str) -> float:
        return sum(e - s for s, e in of(activity))

    def longest(activity: str) -> list[float]:
        return max(of(activity), key=lambda iv: iv[1] - iv[0])

    def at(t: float) -> str | None:
        return next((a for a, s, e in bouts if s <= t < e), None)

    def middle(activity: str) -> float:
        s, e = longest(activity)
        return float(round((s + e) / 2))

    questions: list[dict] = []

    # --- Tier 1: identification and verification ---------------------------
    for n, activity in enumerate(targets["identify"]):
        t = middle(activity)
        assert at(t) == activity
        questions.append(
            _q(
                f"{subject}_t1_id{n}",
                f"What activity is the user performing at {fs(t)} seconds?",
                {"answer": name(activity), "activity_event": name(activity), "question_type": "identification", "answer_kind": "categorical"},
            )
        )

    target = targets["verify"]
    t_hit = middle(target)
    miss = next(c for c in ("SITTING", "LYING", *CANONICAL_CLASSES) if c != at(t_hit))
    for suffix, activity, verdict in (("ver0", target, "Yes"), ("ver1", miss, "No")):
        questions.append(
            _q(
                f"{subject}_t1_{suffix}",
                f"Is the user {phrase(activity)} at {fs(t_hit)} seconds?",
                {"answer": verdict, "activity_event": name(activity), "question_type": "verification", "answer_kind": "categorical"},
            )
        )

    # --- Tier 2: duration, count, comparison -------------------------------
    for n, activity in enumerate(targets["durations"]):
        questions.append(
            _q(
                f"{subject}_t2_dur{n}",
                f"How long was the user {phrase(activity)} in total?",
                {
                    "answer": f"{fs(total(activity))} seconds",
                    "activity_event": name(activity),
                    "question_type": "duration",
                    "answer_kind": "numeric",
                    "numeric_value": total(activity),
                    **_evidence(of(activity)),
                },
            )
        )

    if targets["absent"]:
        absent = targets["absent"]
        assert not of(absent)
        questions.append(
            _q(
                f"{subject}_t2_dur_absent",
                f"How long was the user {phrase(absent)} in total?",
                {
                    "answer": "0 seconds",
                    "activity_event": name(absent),
                    "question_type": "duration",
                    "answer_kind": "numeric",
                    "numeric_value": 0.0,
                },
            )
        )

    counted = targets["count"]
    questions.append(
        _q(
            f"{subject}_t2_count0",
            f"How many separate times was the user {phrase(counted)}?",
            {
                "answer": str(len(of(counted))),
                "activity_event": name(counted),
                "question_type": "count",
                "answer_kind": "numeric",
                "numeric_value": float(len(of(counted))),
                **_evidence(of(counted)),
            },
        )
    )

    for n, (a, b) in enumerate(targets["comparisons"]):
        verdict = "Equal" if total(a) == total(b) else name(a if total(a) > total(b) else b)
        questions.append(
            _q(
                f"{subject}_t2_cmp{n}",
                f"Did the user spend more time {phrase(a)} or {phrase(b)}?",
                {"answer": verdict, "activity_event": f"{name(a)}, {name(b)}", "question_type": "comparison", "answer_kind": "categorical"},
            )
        )

    # --- Tier 3: evidence grounding ----------------------------------------
    for n, activity in enumerate(targets["ground"]):
        questions.append(
            _q(
                f"{subject}_t3_ground{n}",
                f"Cite the stretch of signal where the user was {phrase(activity)}.",
                {
                    "answer": name(activity),
                    "activity_event": name(activity),
                    "question_type": "grounding",
                    "answer_kind": "temporal",
                    **_evidence(of(activity)),
                },
            )
        )

    onset_activity = targets["onset"]
    first = of(onset_activity)[0]
    questions.append(
        _q(
            f"{subject}_t3_onset",
            f"Did the user begin {phrase(onset_activity)} at any point, and if so, when?",
            {
                "answer": f"Yes, {phrase(onset_activity)} began at {fs(first[0])} seconds",
                "activity_event": f"Onset of {phrase(onset_activity)}",
                "question_type": "grounding",
                "answer_kind": "temporal",
                **_evidence([first]),
            },
        )
    )

    # --- Tier 4: open-world reasoning --------------------------------------
    lying = of("LYING")
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

    cycling = of("BICYCLING")
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

    sedentary = sum(total(a) for a in SEDENTARY)
    active = sum(total(a) for a in ACTIVE)
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

    # Edge case: the answer lands inside a real data gap, well away from its
    # edges. It must be a stretch with no sensor data at all, not merely no
    # label, since the oracle also skips recorded minutes nobody labelled.
    # Only the real-model track's window timestamps are read for this -- they
    # are fixed by windowing the raw data -- never its predictions.
    def unrecorded(gap: tuple[float, float]) -> bool:
        return not any(s < gap[1] and e > gap[0] for s, e in recorded)

    gap = next(g for g in gaps if g[1] - g[0] >= 300 and unrecorded(g))
    midpoint = float(round((gap[0] + gap[1]) / 2))
    assert at(midpoint) is None
    questions.append(
        _q(
            f"{subject}_t4_gap",
            f"What was the user doing at {fs(midpoint)} seconds?",
            {"answer": "N/A", "activity_event": "No data", "question_type": "open_world", "answer_kind": "open_world"},
        )
    )
    return questions


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for subject, uuid in UUIDS.items():
        windows = load_track(FIXTURE_DIR / f"track_{subject}.jsonl")
        bouts, gaps = truth_bouts(windows)
        cross_check(subject, windows, bouts, gaps)
        recorded = [(w["t_start"], w["t_end"]) for w in load_track(REAL_TRACK_DIR / f"track_{subject}.jsonl")]
        questions = build_questions(subject, bouts, gaps, recorded)

        payload = {
            "description": (
                f"Dev questions for {subject}, the REAL ExtraSensory subject {uuid}, generated by "
                "scripts/make_real_dev_questions.py from the subject's oracle track "
                "(tests/fixtures/track_{subject}.jsonl). Gold bouts are derived from the ground-truth "
                "labels by an implementation of the docs/TASKS.md section 0 minute-attribution rules that "
                "is independent of ats.aggregate, and verified identical to ats.aggregate.build_timeline "
                "before writing. The oracle is therefore right by construction; differences against the "
                "real-model track are recognition differences. The data-gap question sits in a stretch "
                "with no recorded windows in either track, not merely no labels."
            ).replace("{subject}", subject),
            "questions": questions,
        }
        path = OUT_DIR / f"{subject}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        print(f"{subject}: {len(bouts)} bouts, {len(gaps)} gaps, {len(questions)} questions -> {path.relative_to(REPO_ROOT)}")
        for activity in sorted({a for a, _, _ in bouts}, key=lambda a: -sum(e - s for x, s, e in bouts if x == a)):
            spans = [(s, e) for a, s, e in bouts if a == activity]
            best = max(spans, key=lambda iv: iv[1] - iv[0])
            print(
                f"  {activity:16s} total {fs(sum(e - s for s, e in spans)):>9s} s  bouts {len(spans):4d}  "
                f"longest {fs(best[0])}-{fs(best[1])} ({fs(best[1] - best[0])} s)"
            )


if __name__ == "__main__":
    main()
