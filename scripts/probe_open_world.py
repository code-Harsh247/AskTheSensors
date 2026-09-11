"""Probe questions about behaviours outside the seven classes (docs/TASKS.md
Phase 4 exit criterion): each must get an argued answer from the signal or an
explicit abstention, never the nearest of the seven labels.

These questions were written by Member B, who also wrote the rules they
exercise, so this is a check of intent, not a blind evaluation.

Usage:  python scripts/probe_open_world.py
"""

from __future__ import annotations

from collections import Counter

from ats.aggregate import build_timeline, load_track
from ats.answer import answer_question
from ats.eval.dev import FIXTURES_DIR
from ats.routing import route

QUESTIONS = (
    "Was the user fidgeting?",
    "Was the user restless?",
    "Did the user go for a jog?",
    "Was the user pacing back and forth?",
    "Was the user driving?",
    "Was the user sleeping?",
    "Did the user take a nap?",
    "Was the user doing yoga?",
    "Was the user climbing stairs?",
    "Was the user on a bus or in a car?",
    "Was the user exercising?",
    "Was the user doing chores around the house?",
    "Was the user typing at a desk?",
    "Was the user dancing?",
    "Did the user spend a long stretch without moving?",
    "Was the user commuting?",
)

TRACKS = {
    "subj_real_a oracle": FIXTURES_DIR / "track_subj_real_a.jsonl",
    "subj_real_a real model": FIXTURES_DIR / "real_model_tracks" / "track_subj_real_a.jsonl",
    "subj_real_b oracle": FIXTURES_DIR / "track_subj_real_b.jsonl",
    "subj_real_b real model": FIXTURES_DIR / "real_model_tracks" / "track_subj_real_b.jsonl",
}


def outcome(text: str, answer: dict) -> str:
    call = route(text)
    if answer["answer"] == "N/A":
        return "abstained"
    if call.op == "open_world" and answer["cited_intervals"]:
        return "argued from the signal"
    return f"answered as a named class ({call.op})"


def main() -> None:
    for name, path in TRACKS.items():
        windows = load_track(path)
        timeline = build_timeline(windows)
        tally: Counter[str] = Counter()
        print(f"--- {name}")
        for i, text in enumerate(QUESTIONS):
            answer, _ = answer_question({"question_id": f"p{i}", "text": text}, timeline, windows)
            kind = outcome(text, answer)
            tally[kind] += 1
            print(f"   {text:52s} {answer['answer']:11s} {kind}")
        print("   " + ", ".join(f"{n} {kind}" for kind, n in sorted(tally.items())))


if __name__ == "__main__":
    main()
