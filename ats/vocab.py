"""Shared activity vocabulary: display names, question-phrase lexicon, and the
definitional groupings open-world answers rely on.

Owned by Member B. Vocabulary only, no reasoning logic, so the dev-question
generator can import display names without making gold answers depend on the
code they are meant to test.
"""

from __future__ import annotations

from ats.contracts import CANONICAL_CLASSES

# Matches the class names the PRD uses (§3.1), so answers read the way a
# grader expects rather than as upper-snake identifiers.
DISPLAY_NAMES: dict[str, str] = {
    "LYING": "Lying down",
    "SITTING": "Sitting",
    "STANDING_STILL": "Standing in place",
    "STANDING_MOVING": "Standing and moving",
    "WALKING": "Walking",
    "RUNNING": "Running",
    "BICYCLING": "Bicycling",
}
assert set(DISPLAY_NAMES) == set(CANONICAL_CLASSES)

# Phrases a question may use for each class. Matching is longest-first across
# all classes, so "standing and moving" is never read as bare "standing".
ACTIVITY_PHRASES: dict[str, tuple[str, ...]] = {
    "LYING": ("lying down", "lie down", "lay down", "laid down", "lying", "reclining"),
    "SITTING": ("sitting down", "sitting", "sit", "seated", "sat"),
    "STANDING_STILL": ("standing in place", "standing still", "stand still", "stood still", "standing"),
    "STANDING_MOVING": ("standing and moving", "standing while moving", "moving around"),
    "WALKING": ("walking", "walk", "walked", "strolling"),
    "RUNNING": ("running", "run", "ran", "jogging", "jog", "jogged", "sprinting"),
    "BICYCLING": ("bicycling", "cycling", "biking", "bicycle", "bike", "cycled", "pedaling"),
}
assert set(ACTIVITY_PHRASES) == set(CANONICAL_CLASSES)

# Definitional groupings for open-world questions ("mostly at rest or mostly
# active?", "anything strenuous?"). STANDING_MOVING is light activity and is
# deliberately in neither of the first two groups.
SEDENTARY: tuple[str, ...] = ("LYING", "SITTING", "STANDING_STILL")
ACTIVE: tuple[str, ...] = ("WALKING", "RUNNING", "BICYCLING")
VIGOROUS: tuple[str, ...] = ("RUNNING", "BICYCLING")

# A single continuous stretch at least this long counts as "prolonged" (PRD
# §4.4's "did the user lie down for a prolonged period?"). Five minutes is
# well beyond any brief stationary pause.
PROLONGED_S = 300.0

OPERATOR_TIER: dict[str, int] = {
    "identify": 1,
    "verify": 1,
    "duration": 2,
    "count": 2,
    "compare": 2,
    "onset": 3,
    "ground": 3,
    "open_world": 4,
}


def display(activity: str) -> str:
    return DISPLAY_NAMES[activity]
