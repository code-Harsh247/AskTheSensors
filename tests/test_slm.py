"""The SLM parser is tested without the model: parsing and schema checks run
on hand-written outputs, and the router takes an injected generator. One
test loads the real model, and is skipped unless its pinned weights are
already downloaded."""

import json

import pytest
from _windows import minutes

from ats.aggregate import build_timeline
from ats.answer import answer_all
from ats.routing import route
from ats.slm import MODEL_ID, MODEL_REVISION, SLMRouter, build_messages, parse_operator_call
from ats.validator import grounding_problems


def test_parses_a_well_formed_call():
    call = parse_operator_call('{"op": "compare", "activities": ["WALKING", "RUNNING"], "time_s": null, "predicate": null}')
    assert (call.op, call.activities, call.time_s, call.predicate) == ("compare", ("WALKING", "RUNNING"), None, None)


def test_tolerates_surrounding_text_and_missing_optional_fields():
    call = parse_operator_call('Sure! ```json\n{"op": "verify", "activities": ["RUNNING"], "time_s": 300}\n```')
    assert (call.op, call.activities, call.time_s, call.predicate) == ("verify", ("RUNNING",), 300.0, None)


def test_identify_ignores_any_activities_the_model_adds():
    assert parse_operator_call('{"op": "identify", "activities": ["WALKING"], "time_s": 45}').activities == ()


@pytest.mark.parametrize(
    "raw, reason",
    [
        ("I think the user was walking.", "no JSON object"),
        ('{"op": "duration", "activities": ["WALKING"]', "no JSON object"),
        ('{"op": "summarise", "activities": []}', "schema"),
        ('{"op": "verify", "activities": ["DANCING"]}', "schema"),
        ('{"op": "verify", "activities": ["WALKING"], "time_s": -5}', "schema"),
        ('{"op": "verify", "activities": ["WALKING"], "confidence": 0.9}', "schema"),
        ('{"op": "compare", "activities": ["WALKING"]}', "needs 2 activities"),
        ('{"op": "duration", "activities": []}', "needs 1 activity"),
        ('{"op": "duration", "activities": ["WALKING"], "predicate": "prolonged"}', "only valid for open_world"),
    ],
)
def test_refuses_output_outside_the_closed_schema(raw, reason):
    with pytest.raises(ValueError, match=reason):
        parse_operator_call(raw)


def test_prompt_ends_with_the_question():
    messages = build_messages("How long was the user walking?")
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "How long was the user walking?"}


def test_refused_output_falls_back_to_the_rule_router_and_is_recorded():
    router = SLMRouter(generator=lambda text: "no idea, sorry")
    call, source = router.route("How long was the user walking?")

    assert source == "rules_fallback"
    assert call == route("How long was the user walking?")
    assert (router.n_parsed, router.n_fallback) == (0, 1)
    assert router.failures[0]["reason"].startswith("no JSON object")


def test_accepted_output_is_used_as_is():
    router = SLMRouter(generator=lambda text: '{"op": "count", "activities": ["WALKING"]}')
    call, source = router.route("How often did she walk?")
    assert (source, call.op, call.activities) == ("slm", "count", ("WALKING",))
    assert (router.n_parsed, router.n_fallback) == (1, 0)


def test_a_misparse_can_make_an_answer_wrong_but_never_ungrounded():
    """Whatever operator the model picks, the operators compute the answer
    and the validator gates it."""
    windows = minutes([(0, "SITTING"), (60, "WALKING"), (120, "RUNNING")])
    timeline = build_timeline(windows)
    wrong = SLMRouter(generator=lambda text: '{"op": "duration", "activities": ["RUNNING"]}')

    answers, rejections = answer_all(
        [{"question_id": "q", "text": "How long was the user walking?"}], windows, router=wrong
    )

    assert rejections == []
    assert answers[0]["answer"] == "60 seconds"
    assert answers[0]["cited_intervals"] == [[120.0, 180.0]]
    assert grounding_problems(answers[0], wrong("How long was the user walking?"), timeline) == []


def _model_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    return isinstance(path, str)


@pytest.mark.skipif(not _model_cached(), reason="pinned SLM weights not downloaded")
def test_the_real_model_emits_a_parseable_call():
    router = SLMRouter()
    call, source = router.route("How long was the user walking?")
    assert source == "slm", router.failures
    assert call.op in {"duration", "count", "verify", "ground", "open_world", "identify", "compare", "onset"}
    json.dumps({"op": call.op})
