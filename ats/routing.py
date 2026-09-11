"""Rule-based question routing: natural-language question -> typed operator
call (docs/TASKS.md task 2B.1).

Rules first, deliberately. This is the SLM-independent fallback that keeps
the system answering when a language model misparses, and the baseline a
Phase 4 SLM router has to beat. The router only decides *which* operator to
run and on what; it never produces an answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ats.vocab import ACTIVITY_PHRASES, OPERATOR_TIER


@dataclass(frozen=True)
class OperatorCall:
    op: str
    activities: tuple[str, ...] = ()
    time_s: float | None = None
    predicate: str | None = None

    @property
    def tier(self) -> int:
        return OPERATOR_TIER[self.op]


_PHRASES = sorted(
    ((phrase, cls) for cls, phrases in ACTIVITY_PHRASES.items() for phrase in phrases),
    key=lambda pc: -len(pc[0]),
)


def find_activities(text: str) -> tuple[str, ...]:
    """Canonical classes mentioned in `text`, in order of first mention."""
    lowered = text.lower()
    claimed: list[tuple[int, int]] = []
    hits: list[tuple[int, str]] = []
    for phrase, cls in _PHRASES:
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", lowered):
            start, end = match.span()
            if any(start < c_end and c_start < end for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            hits.append((start, cls))
    ordered: list[str] = []
    for _, cls in sorted(hits):
        if cls not in ordered:
            ordered.append(cls)
    return tuple(ordered)


_TIME_PATTERNS = (
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|secs?|seconds?)\b"), 1.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:mins?|minutes?)\b"), 60.0),
    (re.compile(r"\bt\s*=\s*(\d+(?:\.\d+)?)"), 1.0),
)


def find_time(text: str) -> float | None:
    """A point in time, in seconds from the start of the recording."""
    lowered = text.lower()
    for pattern, scale in _TIME_PATTERNS:
        match = pattern.search(lowered)
        if match:
            return float(match.group(1)) * scale
    return None


_OPEN_WORLD = (
    (
        "activity_balance",
        re.compile(r"\bmostly\b.*\b(rest|active|sedentary)\b|\b(at rest|sedentary)\b.*\bor\b.*\bactive\b"),
    ),
    ("wheeled", re.compile(r"\b(wheeled|wheels?|pedal\w*|mode of (movement|transport\w*))\b")),
    (
        "least_movement",
        re.compile(r"\bleast (movement|active|motion)\b|\bmost (still|stationary|restful)\b|\blowest[- ](movement|activity|intensity)\b"),
    ),
    (
        "most_movement",
        re.compile(r"\bmost (movement|active|motion|intense|vigorous)\b|\bhighest[- ](movement|activity|intensity)\b"),
    ),
    ("prolonged", re.compile(r"\b(prolonged|extended (period|time|stretch)|long (period|stretch|while)|for a long time)\b")),
    ("strenuous", re.compile(r"\b(strenuous|vigorous|intense|exert\w*|exercis\w*|work(ing)? out|workout)\b")),
)
_THRESHOLD = re.compile(r"\b(more|less|longer|shorter|fewer) than\b|\bat (least|most)\b")
_ONSET = re.compile(r"\b(begin|began|begun|start|started|starting|onset|commenc\w*)\b")
_GROUND = re.compile(
    r"\b(cite|citation|evidence|stretch of (the )?signal|where in the (signal|recording)|which part of the (signal|recording)|point to)\b"
)
_COUNT = re.compile(r"\bhow many\b.*\b(times|bouts|episodes|occasions|intervals|periods)\b|\bhow often\b|\bnumber of (times|bouts|episodes)\b")
_DURATION = re.compile(r"\bhow (long|much time)\b|\btotal (time|duration)\b|\bduration\b|\bhow many (seconds|minutes|hours)\b")
_COMPARE = re.compile(r"\b(more|less) time\b|\blonger\b|\bshorter\b|\bcompar\w*\b|\bmore\b")
_VERIFY = re.compile(r"^\s*(is|was|were|did|does|do|has|had)\b")
_IDENTIFY = re.compile(r"\bwhat\b.*\b(activity|doing|performing|up to)\b|\bwhich activity\b")


def route(text: str) -> OperatorCall:
    """Map a question to one operator call. Unrecognised questions become an
    open-world call with no predicate, which the operators answer with an
    explicit abstention rather than a guess."""
    lowered = text.lower()
    activities = find_activities(text)
    time_s = find_time(text)

    for predicate, pattern in _OPEN_WORLD:
        if pattern.search(lowered):
            # A rest question that names no posture ("resting for a long
            # time?") keeps no activity, so it is answered from sustained
            # stillness in the signal rather than from a posture label.
            return OperatorCall("open_world", activities, time_s, predicate)

    # "Did she walk for more than 5 minutes?" is a threshold question, not a
    # point in time. Carrying the quantity makes the duration operator
    # abstain instead of answering whether she was walking at t = 300 s.
    if _THRESHOLD.search(lowered) and activities and time_s is not None:
        return OperatorCall("duration", activities[:1], time_s)
    if _ONSET.search(lowered) and activities:
        return OperatorCall("onset", activities[:1], time_s)
    if _GROUND.search(lowered):
        return OperatorCall("ground", activities[:1], time_s)
    if _COUNT.search(lowered) and activities:
        return OperatorCall("count", activities[:1], time_s)
    if _DURATION.search(lowered) and activities:
        return OperatorCall("duration", activities[:1], time_s)
    if _COMPARE.search(lowered) and len(activities) >= 2:
        return OperatorCall("compare", activities[:2], time_s)
    if _VERIFY.search(lowered) and activities:
        return OperatorCall("verify", activities[:1], time_s)
    if _IDENTIFY.search(lowered):
        return OperatorCall("identify", (), time_s)
    return OperatorCall("open_world", activities, time_s, None)
