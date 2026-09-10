"""Answer I/O: the PRD §5 text block and the machine-scoreable JSONL form.

Owned by Member B (docs/TASKS.md task 1B.2). Every answer is validated
against the frozen schema on the way out, so malformed output cannot reach
a file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from ats.contracts import format_answer_text, validate_answer, validate_question_set

FORMATS = ("text", "jsonl")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def format_seconds(value: float) -> str:
    """Seconds in the frozen time base (3 decimals), trailing zeros dropped:
    600 -> '600', 22.5 -> '22.5'. Never scientific notation, which a
    multi-day recording would otherwise hit."""
    text = f"{round(value, 3):.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def first_number(text: str) -> float | None:
    match = _NUMBER.search(text)
    return float(match.group()) if match else None


def format_intervals(intervals: Iterable[Sequence[float]]) -> str:
    """Render cited intervals, e.g. '905 to 1420, 2110 to 2295 (seconds from start)'."""
    parts = [f"{format_seconds(start)} to {format_seconds(end)}" for start, end in intervals]
    if not parts:
        return "N/A"
    return ", ".join(parts) + " (seconds from start)"


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
