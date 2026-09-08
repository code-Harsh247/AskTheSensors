"""Frozen A<->B contract: canonical constants, JSON Schema validation, and
the PRD §5 text renderer. See docs/TASKS.md §0 and Phase 0 for the freeze
rationale. Changing anything in this file requires joint agreement and its
own dedicated commit.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator  # see docs/CITATIONS.md#jsonschema-python-library

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"

CANONICAL_CLASSES: tuple[str, ...] = (
    "LYING",
    "SITTING",
    "STANDING_STILL",
    "STANDING_MOVING",
    "WALKING",
    "RUNNING",
    "BICYCLING",
)

TIME_BASE = "seconds_from_start"
"""All timestamps in this project are floats, 3 decimal places, seconds
elapsed from the start of the recording (TASKS.md §0)."""


@lru_cache(maxsize=None)
def _load_schema(name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / name
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _validator(name: str) -> Draft202012Validator:
    schema = _load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_window_track(entry: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if `entry` is not a valid window_track record."""
    _validator("window_track.schema.json").validate(entry)


def validate_answer(answer: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if `answer` does not match answer.schema.json."""
    _validator("answer.schema.json").validate(answer)


def validate_question_set(question_set: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if `question_set` does not match question_set.schema.json."""
    _validator("question_set.schema.json").validate(question_set)


def validate_cost_report(report: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if `report` does not match cost_report.schema.json."""
    _validator("cost_report.schema.json").validate(report)


def format_answer_text(answer: dict[str, Any]) -> str:
    """Render a validated answer dict into the exact PRD §5 text block, field
    order fixed: Answer, Activity/Event, Evidence (Timestamp(s), Sensor
    Modality, Sensor Channel(s)), Explanation. ats/serialize.py (Phase 1,
    Member B) builds the fuller machine-format serializer on top of this.
    """
    evidence = answer["evidence"]
    return (
        f"Answer:              {answer['answer']}\n"
        f"Activity/Event:      {answer['activity_event']}\n"
        f"Evidence:\n"
        f"    Timestamp(s):        {evidence['timestamps']}\n"
        f"    Sensor Modality:     {evidence['sensor_modality']}\n"
        f"    Sensor Channel(s):   {evidence['sensor_channels']}\n"
        f"Explanation:         {answer['explanation']}"
    )
