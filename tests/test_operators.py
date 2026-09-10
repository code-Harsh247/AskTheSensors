"""Every operator answer is checked against a hand-built timeline whose
correct answer is obvious, and every answer must also pass the grounding
validator."""

import pytest
from _windows import minutes

from ats.aggregate import build_timeline
from ats.answer import to_answer
from ats.operators import execute
from ats.routing import route
from ats.serialize import first_number
from ats.validator import grounding_problems

# SITTING [0,120)  WALKING [120,300)  gap [300,360)  WALKING [360,420)
# RUNNING [420,540)  LYING [540,840)
PLAN = [
    (0, "SITTING"),
    (60, "SITTING"),
    (120, "WALKING"),
    (180, "WALKING"),
    (240, "WALKING"),
    (360, "WALKING"),
    (420, "RUNNING"),
    (480, "RUNNING"),
    (540, "LYING"),
    (600, "LYING"),
    (660, "LYING"),
    (720, "LYING"),
    (780, "LYING"),
]
WINDOWS = minutes(PLAN)
TIMELINE = build_timeline(WINDOWS)
OBSERVED = [[0.0, 120.0], [120.0, 300.0], [360.0, 420.0], [420.0, 540.0], [540.0, 840.0]]


def ask(text):
    call = route(text)
    answer = to_answer("q", call, execute(call, TIMELINE, WINDOWS))
    assert grounding_problems(answer, call, TIMELINE) == []
    return answer


def test_timeline_under_test():
    assert [(iv.activity, iv.t_start, iv.t_end) for iv in TIMELINE.intervals] == [
        ("SITTING", 0.0, 120.0),
        ("WALKING", 120.0, 300.0),
        ("WALKING", 360.0, 420.0),
        ("RUNNING", 420.0, 540.0),
        ("LYING", 540.0, 840.0),
    ]
    assert TIMELINE.gaps == ((300.0, 360.0),)


# --- Tier 1 -----------------------------------------------------------------


def test_identify_at_a_time():
    answer = ask("What activity is the user performing at 130 seconds?")
    assert answer["answer"] == "Walking"
    assert answer["cited_intervals"] == [[120.0, 300.0]]
    assert answer["tier_inferred"] == 1


def test_identify_inside_a_gap_abstains_and_names_the_gap():
    answer = ask("What was the user doing at 330 seconds?")
    assert answer["answer"] == "N/A"
    assert "300" in answer["explanation"] and "360" in answer["explanation"]


def test_identify_outside_the_recording_abstains():
    assert ask("What activity is the user performing at 5000 seconds?")["answer"] == "N/A"


def test_identify_without_a_time_reports_the_dominant_activity():
    answer = ask("What activity is the user performing?")
    assert answer["answer"] == "Lying down"
    assert answer["cited_intervals"] == [[540.0, 840.0]]


def test_verify_at_a_time():
    assert ask("Is the user running at 430 seconds?")["answer"] == "Yes"
    answer = ask("Is the user running at 130 seconds?")
    assert answer["answer"] == "No"
    assert answer["cited_intervals"] == [[120.0, 300.0]]


def test_verify_an_absent_activity_cites_everything_examined():
    answer = ask("Is the user bicycling?")
    assert answer["answer"] == "No"
    assert answer["cited_intervals"] == OBSERVED


# --- Tier 2 -----------------------------------------------------------------


def test_duration_sums_non_contiguous_bouts():
    answer = ask("How long was the user walking?")
    assert answer["answer"] == "240 seconds"
    assert answer["cited_intervals"] == [[120.0, 300.0], [360.0, 420.0]]


def test_duration_explanation_is_built_from_the_windows_behind_it():
    explanation = ask("How long was the user walking?")["explanation"]
    assert "40 windows" in explanation
    assert "88 s of recorded signal" in explanation


def test_duration_of_an_absent_activity_is_zero():
    assert ask("How long was the user bicycling?")["answer"] == "0 seconds"


def test_count_counts_bouts_not_bursts():
    assert first_number(ask("How many separate times was the user walking?")["answer"]) == 2


def test_compare():
    assert ask("Did the user spend more time walking or sitting?")["answer"] == "Walking"


def test_compare_tie():
    assert ask("Did the user spend more time sitting or running?")["answer"] == "Equal"


def test_time_restricted_duration_abstains_rather_than_answering_a_different_question():
    assert ask("How long was the user walking at 130 seconds?")["answer"] == "N/A"
    assert ask("Did the user walk for more than 5 minutes?")["answer"] == "N/A"


# --- Tier 3 -----------------------------------------------------------------


def test_onset_cites_the_first_bout_and_the_transition_into_it():
    answer = ask("Did the user begin running at any point, and if so, when?")
    assert answer["answer"] == "Yes, running began at 420 seconds"
    assert answer["cited_intervals"] == [[420.0, 540.0]]
    assert "walking" in answer["explanation"]
    assert answer["tier_inferred"] == 3


def test_onset_of_the_first_activity_in_the_recording_says_so():
    answer = ask("Did the user begin sitting at any point?")
    assert answer["answer"] == "Yes, sitting began at 0 seconds"
    assert "first interval in the recording" in answer["explanation"]


def test_onset_after_a_gap_warns_that_the_true_onset_may_be_earlier():
    windows = minutes([(0, "SITTING"), (120, "WALKING")])
    timeline = build_timeline(windows)
    call = route("When did the user start walking?")
    finding = execute(call, timeline, windows)
    assert "gap" in finding.explanation


def test_ground():
    answer = ask("Cite the stretch of signal where the user was sitting.")
    assert answer["answer"] == "Sitting"
    assert answer["cited_intervals"] == [[0.0, 120.0]]


# --- Tier 4 -----------------------------------------------------------------


def test_prolonged_rest():
    answer = ask("Did the user lie down for a prolonged period?")
    assert answer["answer"] == "Likely yes"
    assert answer["cited_intervals"] == [[540.0, 840.0]]


def test_wheeled_movement_absent():
    answer = ask("Was the user using a wheeled or pedal-based mode of movement?")
    assert answer["answer"] == "No"
    assert answer["cited_intervals"] == OBSERVED


def test_activity_balance():
    # sedentary: sitting 120 + lying 300 = 420; active: walking 240 + running 120 = 360
    assert ask("Was the user mostly at rest or mostly active?")["answer"] == "Mostly at rest"


def test_least_and_most_movement_are_read_from_the_signal():
    assert ask("Which stretch of the recording shows the least movement?")["answer"] == "Lying down"
    assert ask("Which stretch shows the most movement?")["answer"] == "Running"


def test_strenuous():
    answer = ask("Was the user doing anything strenuous?")
    assert answer["answer"] == "Yes"
    assert answer["cited_intervals"] == [[420.0, 540.0]]


def test_unrecognised_question_abstains():
    answer = ask("Tell me something interesting.")
    assert answer["answer"] == "N/A"
    assert answer["cited_intervals"] == []


@pytest.mark.parametrize(
    "text",
    [
        "What activity is the user performing?",
        "How long was the user lying down?",
        "Did the user spend more time walking or running?",
        "Did the user begin lying down at any point?",
        "Was the user mostly at rest or mostly active?",
    ],
)
def test_evidence_fields_are_consistent(text):
    answer = ask(text)
    assert answer["modality"] == "both"
    assert answer["channels"] == ["all"]
    assert answer["evidence"]["sensor_modality"] == "Accelerometer, Gyroscope"
