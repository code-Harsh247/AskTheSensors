"""Grounding validator (docs/TASKS.md task 2B.4): the hard gate between a
computed answer and the output file. It is the architectural enforcement of
PRD §1.3 -- an answer not tied to the recorded signal does not meet the
requirement -- and it checks the answer dict itself, so it catches a
fabrication whether it came from a bug or, later, from a language model.

An explicit abstention (answer "N/A") makes no claim, so it needs no
evidence. Anything else must cite intervals that exist in the timeline and
state numbers that survive recomputation.
"""

from __future__ import annotations

from typing import Any

from ats.aggregate import Timeline
from ats.routing import OperatorCall
from ats.serialize import first_number, format_intervals, format_seconds

_EPS = 1e-6
# Operators render seconds to 3 decimals, so a faithful restatement differs
# from the recomputed value by at most half a millisecond.
_NUMERIC_EPS = 1e-3


class UngroundedAnswerError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def is_abstention(answer: dict[str, Any]) -> bool:
    return answer["answer"].strip().upper() == "N/A"


def grounding_problems(answer: dict[str, Any], call: OperatorCall, timeline: Timeline) -> list[str]:
    problems: list[str] = []
    cited = [tuple(span) for span in answer["cited_intervals"]]

    if answer["evidence"]["timestamps"] != format_intervals(cited):
        problems.append("evidence timestamps text does not match cited_intervals")

    for start, end in cited:
        if not any(iv.t_start - _EPS <= start and end <= iv.t_end + _EPS for iv in timeline.intervals):
            problems.append(
                f"cited interval {format_seconds(start)}-{format_seconds(end)} s is not inside any timeline interval"
            )

    if is_abstention(answer):
        return problems

    if call.tier >= 3 and not cited:
        problems.append(f"tier-{call.tier} answer asserts a finding without citing any evidence")

    if call.op in ("duration", "count"):
        activity = call.activities[0]
        expected = (
            timeline.total_duration(activity) if call.op == "duration" else float(timeline.count(activity))
        )
        stated = first_number(answer["answer"])
        if stated is None:
            problems.append("numeric answer does not state a number")
        elif abs(stated - expected) > _NUMERIC_EPS:
            problems.append(
                f"stated {format_seconds(stated)} disagrees with {format_seconds(expected)} recomputed from the timeline"
            )

    return problems


def validate_grounding(answer: dict[str, Any], call: OperatorCall, timeline: Timeline) -> None:
    problems = grounding_problems(answer, call, timeline)
    if problems:
        raise UngroundedAnswerError(problems)
