"""The grounding validator must reject fabricated answers (docs/TASKS.md
Phase 2 exit criterion) and the pipeline must never emit one."""

import pytest
from _windows import minutes

from ats.aggregate import build_timeline
from ats.answer import answer_question, to_answer
from ats.operators import Finding, abstain
from ats.routing import route
from ats.validator import UngroundedAnswerError, grounding_problems, validate_grounding

# SITTING [0,60)  WALKING [60,180)  gap [180,240)  WALKING [240,300)  RUNNING [300,360)
WINDOWS = minutes(
    [(0, "SITTING"), (60, "WALKING"), (120, "WALKING"), (240, "WALKING"), (300, "RUNNING")]
)
TIMELINE = build_timeline(WINDOWS)
DURATION = route("How long was the user walking?")
ONSET = route("Did the user begin running at any point, and if so, when?")
WALKING_SPANS = ((60.0, 180.0), (240.0, 300.0))


def check(call, finding):
    return grounding_problems(to_answer("q", call, finding), call, TIMELINE)


def test_timeline_under_test():
    assert TIMELINE.total_duration("WALKING") == pytest.approx(180.0)
    assert TIMELINE.gaps == ((180.0, 240.0),)


def test_genuine_answer_passes():
    assert check(DURATION, Finding("180 seconds", "Walking", WALKING_SPANS, "...")) == []


def test_interval_absent_from_the_timeline_is_rejected():
    problems = check(DURATION, Finding("180 seconds", "Walking", ((190.0, 230.0),), "..."))
    assert any("not inside any timeline interval" in p for p in problems)


def test_duration_off_by_30_seconds_is_rejected():
    problems = check(DURATION, Finding("210 seconds", "Walking", WALKING_SPANS, "..."))
    assert any("disagrees" in p for p in problems)


def test_count_that_disagrees_with_the_timeline_is_rejected():
    count = route("How many separate times was the user walking?")
    problems = check(count, Finding("3 separate bouts", "Walking", WALKING_SPANS, "..."))
    assert any("disagrees" in p for p in problems)


def test_tier3_claim_without_evidence_is_rejected():
    problems = check(ONSET, Finding("Yes, running began at 300 seconds", "Onset of running", (), "..."))
    assert any("without citing any evidence" in p for p in problems)


def test_timestamps_text_must_match_the_cited_intervals():
    answer = to_answer("q", DURATION, Finding("180 seconds", "Walking", WALKING_SPANS, "..."))
    answer["evidence"]["timestamps"] = "60 to 999 (seconds from start)"
    assert "evidence timestamps text does not match cited_intervals" in grounding_problems(
        answer, DURATION, TIMELINE
    )


def test_an_abstention_needs_no_evidence_even_at_tier_3():
    assert check(ONSET, abstain("cannot tell")) == []


def test_validate_grounding_raises_with_every_problem():
    answer = to_answer("q", DURATION, Finding("210 seconds", "Walking", ((190.0, 230.0),), "..."))
    with pytest.raises(UngroundedAnswerError) as excinfo:
        validate_grounding(answer, DURATION, TIMELINE)
    assert len(excinfo.value.problems) == 2


def test_a_rejected_answer_is_never_emitted(monkeypatch):
    fabricated = Finding("999 seconds", "Walking", WALKING_SPANS, "made up")
    monkeypatch.setattr("ats.answer.execute", lambda call, timeline, windows: fabricated)

    answer, problems = answer_question(
        {"question_id": "q", "text": "How long was the user walking?"}, TIMELINE, WINDOWS
    )

    assert problems
    assert answer["answer"] == "N/A"
    assert answer["cited_intervals"] == []
    assert answer["explanation"].startswith("Withheld")


@pytest.mark.parametrize(
    "text",
    [
        "How long was the user walking?",
        "How many separate times was the user walking?",
        "Did the user begin running at any point, and if so, when?",
        "Cite the stretch of signal where the user was walking.",
        "Is the user bicycling?",
        "What was the user doing at 200 seconds?",
    ],
)
def test_real_operator_output_is_never_withheld(text):
    _, problems = answer_question({"question_id": "q", "text": text}, TIMELINE, WINDOWS)
    assert problems == []
