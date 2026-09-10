import pytest

from ats.routing import find_activities, find_time, route


@pytest.mark.parametrize(
    "text, op, activities, time_s, predicate",
    [
        # The brief's own example questions (PRD §4).
        ("What activity is the user performing?", "identify", (), None, None),
        ("Is the user running?", "verify", ("RUNNING",), None, None),
        ("How long was the user walking?", "duration", ("WALKING",), None, None),
        ("Did the user spend more time walking or running?", "compare", ("WALKING", "RUNNING"), None, None),
        ("Did the user begin running at any point, and if so, when?", "onset", ("RUNNING",), None, None),
        ("Did the user lie down for a prolonged period?", "open_world", ("LYING",), None, "prolonged"),
        ("Was the user using a wheeled or pedal-based mode of movement?", "open_world", (), None, "wheeled"),
        # Dev-set templates.
        ("What activity is the user performing at 220 seconds?", "identify", (), 220.0, None),
        ("Is the user standing and moving at 30 seconds?", "verify", ("STANDING_MOVING",), 30.0, None),
        ("How many separate times did the user do walking?", "count", ("WALKING",), None, None),
        ("How long was the user sitting in total?", "duration", ("SITTING",), None, None),
        ("Cite the stretch of signal where the user was walking.", "ground", ("WALKING",), None, None),
        ("Was the user mostly at rest or mostly physically active during this recording?", "open_world", (), None, "activity_balance"),
        ("Which single stretch of the recording shows the least movement, and why?", "open_world", (), None, "least_movement"),
        ("What was the user doing at 340 seconds?", "identify", (), 340.0, None),
        # The scenario's own phrasing (PRD §1.1).
        ("Was she doing anything strenuous?", "open_world", (), None, "strenuous"),
        ("How much time did the user spend lying down?", "duration", ("LYING",), None, None),
        # A threshold is carried as a quantity, so the operator can refuse it.
        ("Did the user walk for more than 5 minutes?", "duration", ("WALKING",), 300.0, None),
    ],
)
def test_route(text, op, activities, time_s, predicate):
    call = route(text)
    assert call.op == op
    assert call.activities == activities
    assert call.time_s == time_s
    assert call.predicate == predicate


def test_unrecognised_question_routes_to_abstaining_open_world():
    call = route("Tell me something interesting.")
    assert call.op == "open_world"
    assert call.predicate is None


def test_longest_phrase_wins_over_its_prefix():
    assert find_activities("standing and moving, then standing still") == (
        "STANDING_MOVING",
        "STANDING_STILL",
    )


def test_word_boundaries_prevent_substring_matches():
    assert find_activities("Was that likely a runtime error?") == ()


def test_activities_keep_mention_order():
    assert find_activities("more time running or walking?") == ("RUNNING", "WALKING")


def test_time_units():
    assert find_time("at 220 seconds") == 220.0
    assert find_time("at 5 minutes") == 300.0
    assert find_time("at t=12.5") == 12.5
    assert find_time("how long overall") is None


def test_tiers():
    assert route("Is the user running?").tier == 1
    assert route("How long was the user walking?").tier == 2
    assert route("Did the user begin running at any point?").tier == 3
    assert route("Did the user lie down for a prolonged period?").tier == 4
