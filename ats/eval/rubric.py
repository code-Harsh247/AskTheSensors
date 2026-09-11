"""Explanation-quality rubric (docs/TASKS.md task 4B.4, PRD §7.3.5).

PRD §7.3.5 has open-world explanations judged for faithfulness and
plausibility on a 1-5 scale, by human graders or by an LLM judge with a fixed
rubric. We use an LLM judge (Claude Haiku 4.5, docs/CITATIONS.md#claude-haiku-45), and
report its consistency as agreement between two independent runs over the
same sample, as PRD §11 asks of whichever grading we pick.

The judge is shown what the system measured for the cited intervals, so it
can check the explanation's numbers against the signal rather than taking
them on trust. It scores; it never produces an answer, and nothing it says
reaches the system's output.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Sequence

from ats.aggregate import Timeline, build_timeline
from ats.answer import answer_all
from ats.evidence import summarize
from ats.signal import STILL_ACC_STD, STILL_GYRO_ENERGY

CRITERIA = ("cites_real_features", "features_support_conclusion", "conclusion_plausible")
SCORES = (1, 2, 3, 4, 5)

RUBRIC = f"""You grade explanations produced by a system that answers questions about a person's activity from a phone's accelerometer and gyroscope. Each explanation justifies an answer to a free-form question. You are given the question, the answer, the explanation, and the values the system actually measured over the stretches of signal the answer cites.

Score the explanation from 1 (worst) to 5 (best) on each criterion:

1. cites_real_features: does the explanation cite concrete signal features (accelerometer magnitude and its variability, gyroscope energy, cadence, share of still windows, durations, time intervals), and do the cited values match the measured values given? 1 = no signal features, or values that contradict the measurements; 5 = specific features whose values all match.
2. features_support_conclusion: do the cited features actually support the stated answer? 1 = the features contradict or are irrelevant to the answer; 5 = the features directly establish it.
3. conclusion_plausible: is the answer a plausible reading of the question given the evidence, including any stated limitation of what the sensors can show? 1 = implausible or overclaimed; 5 = plausible and appropriately hedged.

For reference, the system calls a 4-second window still when accelerometer-magnitude standard deviation is at most {STILL_ACC_STD} m/s^2 and gyroscope energy is at most {STILL_GYRO_ENERGY} above the recording's resting level. Accelerometer magnitude near 9.8 m/s^2 is gravity.

Judge only what is given. Keep the rationale to two sentences."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{criterion: {"type": "integer", "enum": list(SCORES)} for criterion in CRITERIA},
        "rationale": {"type": "string"},
    },
    "required": [*CRITERIA, "rationale"],
    "additionalProperties": False,
}


def build_items(
    sources: Sequence[tuple[str, Sequence[dict[str, Any]], Sequence[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], int]:
    """The explanations to judge: every tier-4 answer that makes a claim,
    from each (label, windows, questions) source, with the measured values
    behind it. An abstention makes no claim and has nothing to judge, so
    abstentions are counted and set aside. Identical explanations to the same
    question are judged once."""
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    abstained = 0
    for label, windows, questions in sources:
        timeline: Timeline = build_timeline(windows)
        answers, _ = answer_all(questions, windows)
        texts = {q["question_id"]: q["text"] for q in questions}
        for answer in answers:
            if answer["tier_inferred"] != 4:
                continue
            if answer["answer"] == "N/A":
                abstained += 1
                continue
            question = texts[answer["question_id"]]
            if (question, answer["explanation"]) in seen:
                continue
            seen.add((question, answer["explanation"]))
            summary = summarize(windows, [tuple(span) for span in answer["cited_intervals"]])
            span = timeline.span()
            items.append(
                {
                    "item_id": f"{label}:{answer['question_id']}",
                    "question": question,
                    "answer": answer["answer"],
                    "activity_event": answer["activity_event"],
                    "explanation": answer["explanation"],
                    "n_cited_intervals": len(answer["cited_intervals"]),
                    "recording_span_s": span[1] - span[0] if span else 0.0,
                    "measured": asdict(summary) if summary else None,
                }
            )
    return items, abstained


def judge_prompt(item: dict[str, Any]) -> str:
    shown = {
        key: item[key]
        for key in ("question", "answer", "activity_event", "explanation", "n_cited_intervals", "recording_span_s", "measured")
    }
    return "Grade this explanation.\n\n" + json.dumps(shown, indent=2, sort_keys=True)


def parse_judgment(text: str) -> dict[str, Any]:
    """The judge's JSON, checked: every criterion present and on the 1-5
    scale. Anything else raises rather than becoming a score."""
    data = json.loads(text)
    for criterion in CRITERIA:
        if data.get(criterion) not in SCORES:
            raise ValueError(f"judge returned {data.get(criterion)!r} for {criterion}")
    return {criterion: data[criterion] for criterion in CRITERIA} | {"rationale": str(data.get("rationale", ""))}


def weighted_kappa(a: Sequence[int], b: Sequence[int], levels: Sequence[int] = SCORES) -> float | None:
    """Quadratic-weighted Cohen's kappa between two ratings of the same items.
    None when it is undefined: every rating identical on both sides leaves no
    variation to agree beyond chance about."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty rating lists")
    k = len(levels)
    index = {level: i for i, level in enumerate(levels)}
    n = len(a)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[index[x]][index[y]] += 1.0 / n
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def weight(i: int, j: int) -> float:
        return (i - j) ** 2 / (k - 1) ** 2

    disagreement = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    expected = sum(weight(i, j) * row[i] * col[j] for i in range(k) for j in range(k))
    if expected == 0.0:
        return None
    return 1.0 - disagreement / expected


def summarize_runs(run_a: Sequence[dict[str, Any]], run_b: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean score per criterion and the two runs' agreement, over the items
    both runs scored."""
    scored_b = {r["item_id"]: r for r in run_b}
    pairs = [(r, scored_b[r["item_id"]]) for r in run_a if r["item_id"] in scored_b]
    if not pairs:
        raise ValueError("no item was scored by both runs")
    report: dict[str, Any] = {"n_items": len(pairs), "criteria": {}}
    for criterion in CRITERIA:
        a = [x[criterion] for x, _ in pairs]
        b = [y[criterion] for _, y in pairs]
        report["criteria"][criterion] = {
            "mean": (sum(a) + sum(b)) / (2 * len(pairs)),
            "exact_agreement": sum(x == y for x, y in zip(a, b)) / len(pairs),
            "within_one": sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(pairs),
            "weighted_kappa": weighted_kappa(a, b),
        }
    report["overall_mean"] = sum(c["mean"] for c in report["criteria"].values()) / len(CRITERIA)
    return report
