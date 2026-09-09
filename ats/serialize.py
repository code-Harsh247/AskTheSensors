"""Answer I/O: the PRD §5 text block and the machine-scoreable JSONL form.

Owned by Member B (docs/TASKS.md task 1B.2). Every answer is validated
against the frozen schema on the way out, so malformed output cannot reach
a file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from ats.contracts import format_answer_text, validate_answer, validate_question_set

FORMATS = ("text", "jsonl")


def to_text(answer: dict[str, Any], query: str | None = None) -> str:
    """Render one answer as the PRD §5 block, optionally preceded by the query
    line the brief's own examples show."""
    validate_answer(answer)
    block = format_answer_text(answer)
    return f'Query: "{query}"\n{block}' if query is not None else block


def to_jsonl_line(answer: dict[str, Any]) -> str:
    validate_answer(answer)
    return json.dumps(answer, ensure_ascii=False, sort_keys=True)


def write_answers(
    answers: Sequence[dict[str, Any]],
    path: str | Path,
    fmt: str = "text",
    queries: dict[str, str] | None = None,
) -> None:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        if fmt == "jsonl":
            for answer in answers:
                f.write(to_jsonl_line(answer) + "\n")
        else:
            blocks = [
                to_text(a, (queries or {}).get(a["question_id"])) for a in answers
            ]
            f.write("\n\n".join(blocks) + "\n")


def read_answers_jsonl(path: str | Path) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                answer = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
            validate_answer(answer)
            answers.append(answer)
    return answers


def read_question_set(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        question_set = json.load(f)
    validate_question_set(question_set)
    return question_set


def na_answer(question_id: str, tier: int = 1) -> dict[str, Any]:
    """A well-formed answer that declines to answer. Preferred over emitting a
    confident guess when the timeline cannot support one."""
    return {
        "question_id": question_id,
        "answer": "N/A",
        "activity_event": "N/A",
        "evidence": {
            "timestamps": "N/A",
            "sensor_modality": "N/A",
            "sensor_channels": "N/A",
        },
        "explanation": "N/A",
        "tier_inferred": tier,
        "cited_intervals": [],
        "modality": "N/A",
        "channels": ["N/A"],
    }


def format_intervals(intervals: Iterable[tuple[float, float]]) -> str:
    """Render cited intervals in the frozen time base, e.g.
    '905 to 1420, 2110 to 2295 (seconds from start)'."""
    parts = [f"{start:g} to {end:g}" for start, end in intervals]
    if not parts:
        return "N/A"
    return ", ".join(parts) + " (seconds from start)"
