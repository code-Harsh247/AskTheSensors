"""SLM question parser (docs/TASKS.md tasks 4B.1-4B.2): natural language ->
a typed operator call, validated against a closed schema, with the rule
router as the automatic fallback.

The model's only output is an operator call. It never sees the recording and
never produces a number, an interval, or a verdict; the deterministic
operators still compute every answer and the grounding validator still gates
it, so a misparse can make an answer wrong but never ungrounded.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from jsonschema import Draft202012Validator

from ats.contracts import CANONICAL_CLASSES
from ats.routing import OperatorCall, route
from ats.vocab import OPERATOR_TIER

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # see docs/CITATIONS.md#qwen25-05b-instruct
# Pinned so a later upload to the same repository cannot silently change results.
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_NEW_TOKENS = 64

PREDICATES = ("prolonged", "wheeled", "activity_balance", "least_movement", "most_movement", "strenuous")

CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["op"],
    "additionalProperties": False,
    "properties": {
        "op": {"enum": list(OPERATOR_TIER)},
        "activities": {
            "type": "array",
            "items": {"enum": list(CANONICAL_CLASSES)},
            "maxItems": 2,
            "uniqueItems": True,
        },
        "time_s": {"type": ["number", "null"], "minimum": 0},
        "predicate": {"enum": [*PREDICATES, None]},
    },
}
_VALIDATOR = Draft202012Validator(CALL_SCHEMA)
_ACTIVITIES_NEEDED = {"verify": 1, "duration": 1, "count": 1, "onset": 1, "ground": 1, "compare": 2}

SYSTEM_PROMPT = """You convert a question about a person's physical activity into one JSON operator call. Reply with the JSON object only.

Operators:
- identify: what the person was doing, optionally at a time
- verify: whether the person was doing one activity, optionally at a time
- duration: how long one activity lasted in total
- count: how many separate times one activity happened
- compare: which of two activities took more time
- onset: whether and when one activity began
- ground: where in the signal one activity happened
- open_world: anything else, with predicate prolonged, wheeled, activity_balance, least_movement, most_movement, strenuous, or null if none fits

Activities: LYING, SITTING, STANDING_STILL, STANDING_MOVING, WALKING, RUNNING, BICYCLING.
Fields: {"op": ..., "activities": [...], "time_s": seconds from the start of the recording or null, "predicate": ... or null}"""

# Phrasings deliberately unlike the dev-set templates, so the dev set measures
# generalisation rather than recall of these examples.
FEW_SHOT: tuple[tuple[str, dict[str, Any]], ...] = (
    ("At the 45 second mark, what was going on?", {"op": "identify", "activities": [], "time_s": 45, "predicate": None}),
    ("Was he jogging at 300 s?", {"op": "verify", "activities": ["RUNNING"], "time_s": 300, "predicate": None}),
    ("Total time spent seated?", {"op": "duration", "activities": ["SITTING"], "time_s": None, "predicate": None}),
    ("How often did she go for a walk?", {"op": "count", "activities": ["WALKING"], "time_s": None, "predicate": None}),
    ("Was there more cycling than running?", {"op": "compare", "activities": ["BICYCLING", "RUNNING"], "time_s": None, "predicate": None}),
    ("When did the person first get on the bike?", {"op": "onset", "activities": ["BICYCLING"], "time_s": None, "predicate": None}),
    ("Show me where in the data she is lying in bed.", {"op": "ground", "activities": ["LYING"], "time_s": None, "predicate": None}),
    ("Did he rest in bed for a long stretch?", {"op": "open_world", "activities": ["LYING"], "time_s": None, "predicate": "prolonged"}),
    ("Was the person mostly sedentary?", {"op": "open_world", "activities": [], "time_s": None, "predicate": "activity_balance"}),
    ("What is the capital of France?", {"op": "open_world", "activities": [], "time_s": None, "predicate": None}),
)

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def build_messages(question: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example, call in FEW_SHOT:
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": json.dumps(call)})
    messages.append({"role": "user", "content": question})
    return messages


def parse_operator_call(raw: str) -> OperatorCall:
    """Turn model output into an OperatorCall, or raise ValueError naming why
    it was refused."""
    match = _JSON_OBJECT.search(raw)
    if match is None:
        raise ValueError("no JSON object in the model output")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    data.setdefault("activities", [])
    data.setdefault("time_s", None)
    data.setdefault("predicate", None)

    error = next(iter(sorted(_VALIDATOR.iter_errors(data), key=str)), None)
    if error is not None:
        raise ValueError(f"schema: {error.message}")

    op = data["op"]
    activities = tuple(data["activities"])
    needed = _ACTIVITIES_NEEDED.get(op, 0)
    if len(activities) < needed:
        raise ValueError(f"{op} needs {needed} activit{'y' if needed == 1 else 'ies'}")
    if data["predicate"] is not None and op != "open_world":
        raise ValueError("a predicate is only valid for open_world")

    if op == "identify":
        activities = ()
    elif needed:
        activities = activities[:needed]
    time_s = None if data["time_s"] is None else float(data["time_s"])
    return OperatorCall(op, activities, time_s, data["predicate"])


class SLMRouter:
    """Routes with the language model, falling back to the rule router on any
    output it refuses. Every fallback is counted and kept with its reason, so
    the ablation reports how often the model actually decided."""

    def __init__(self, generator: Callable[[str], str] | None = None):
        self._generator = generator
        self.n_parsed = 0
        self.n_fallback = 0
        self.failures: list[dict[str, str]] = []

    @staticmethod
    def load_model() -> Callable[[str], str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32)
        model.eval()
        # Greedy decoding with no sampling knobs, so the same question always
        # gets the same parse.
        config = GenerationConfig(
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        def generate(question: str) -> str:
            inputs = tokenizer.apply_chat_template(
                build_messages(question), add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
            with torch.no_grad():
                output = model.generate(**inputs, generation_config=config)
            return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        return generate

    def route(self, text: str) -> tuple[OperatorCall, str]:
        if self._generator is None:
            self._generator = self.load_model()
        raw = self._generator(text)
        try:
            call = parse_operator_call(raw)
        except ValueError as exc:
            self.n_fallback += 1
            self.failures.append({"question": text, "output": raw, "reason": str(exc)})
            return route(text), "rules_fallback"
        self.n_parsed += 1
        return call, "slm"

    def __call__(self, text: str) -> OperatorCall:
        return self.route(text)[0]
