"""Oracle-vs-real delta (docs/TASKS.md task 3.2).

Runs the identical question set through a subject's oracle track (ground-truth
labels) and its real track (the trained model), then attributes every wrong
answer to the layer that caused it:

- recognition: right on the oracle, wrong on the real track. Only the
  classifier's output changed, so the classifier owns it (Member A).
- routing: wrong even on the oracle, and the router picked an operator that
  cannot answer this kind of question (Member B).
- reasoning: wrong even on the oracle despite a suitable operator, so
  aggregation or the operator itself is at fault (Member B).

An answer wrong on the oracle but right on the real track is a masked bug:
not a loss today, but a routing or reasoning defect that recognition noise
happened to hide.

"Right" means correct and, where the gold cites evidence, also grounded
(PRD §7.3.4), so a regression in evidence alone is attributed too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ats.aggregate import load_track
from ats.answer import answer_all
from ats.eval import QUESTION_TYPES, score_answer
from ats.eval.dev import FIXTURES_DIR, QUESTIONS_DIR, dev_subjects, perturb_bursts, perturb_labels
from ats.eval.metrics import is_grounded_and_correct
from ats.routing import route
from ats.serialize import read_question_set

LAYERS = ("recognition", "routing", "reasoning")
OWNER = {"recognition": "Member A", "routing": "Member B", "reasoning": "Member B"}

COMPATIBLE_OPERATORS: dict[str, frozenset[str]] = {
    "identification": frozenset({"identify"}),
    "verification": frozenset({"verify"}),
    "duration": frozenset({"duration"}),
    "count": frozenset({"count"}),
    "comparison": frozenset({"compare"}),
    "grounding": frozenset({"ground", "onset"}),
    # "What was the user doing at t?" asked about a data gap is gold-typed
    # open-world, and identify answers it correctly.
    "open_world": frozenset({"open_world", "identify"}),
}


def is_right(answer: dict[str, Any], gold: dict[str, Any]) -> bool:
    correct = score_answer(answer, gold)
    if not gold.get("cited_intervals"):
        return correct
    return is_grounded_and_correct(answer, gold, correct)


def attribute(oracle_right: bool, real_right: bool, routed_ok: bool) -> tuple[str | None, bool]:
    """(layer, is_loss). The layer is None when both tracks answer right."""
    if oracle_right:
        return (None, False) if real_right else ("recognition", True)
    return ("reasoning" if routed_ok else "routing", not real_right)


def compare_subject(
    subject: str,
    questions: Sequence[dict[str, Any]],
    oracle_windows: Sequence[dict[str, Any]],
    real_windows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    oracle_answers, oracle_withheld = answer_all(questions, oracle_windows)
    real_answers, real_withheld = answer_all(questions, real_windows)

    rows: list[dict[str, Any]] = []
    for question, oracle, real in zip(questions, oracle_answers, real_answers):
        gold = question.get("gold")
        if not gold:
            continue
        operator = route(question["text"]).op
        routed_ok = operator in COMPATIBLE_OPERATORS.get(gold.get("question_type", ""), frozenset())
        oracle_right, real_right = is_right(oracle, gold), is_right(real, gold)
        layer, is_loss = attribute(oracle_right, real_right, routed_ok)
        rows.append(
            {
                "subject": subject,
                "question_id": question["question_id"],
                "question_type": gold.get("question_type"),
                "operator": operator,
                "oracle_right": oracle_right,
                "real_right": real_right,
                "layer": layer,
                "is_loss": is_loss,
                "gold_answer": gold.get("answer"),
                "oracle_answer": oracle["answer"],
                "real_answer": real["answer"],
            }
        )
    return rows, {"oracle": len(oracle_withheld), "real": len(real_withheld)}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = [(t, [r for r in rows if r["question_type"] == t]) for t in QUESTION_TYPES]
    groups.append(("overall", list(rows)))
    table: dict[str, dict[str, Any]] = {}
    for name, group in groups:
        if not group:
            continue
        table[name] = {
            "n": len(group),
            "oracle_right": sum(r["oracle_right"] for r in group),
            "real_right": sum(r["real_right"] for r in group),
            "losses": {
                layer: sum(1 for r in group if r["is_loss"] and r["layer"] == layer) for layer in LAYERS
            },
            "masked_bugs": sum(1 for r in group if r["layer"] and not r["is_loss"]),
        }
    return table


def labelled_windows(
    real_windows: Sequence[dict[str, Any]], oracle_windows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Real-model windows the oracle also covers. The oracle skips recorded
    minutes nobody labelled, so there is no truth there: a prediction on them
    is neither right nor wrong, and scoring it would blame recognition for a
    difference in coverage."""
    covered = {(round(w["t_start"], 3), round(w["t_end"], 3)) for w in oracle_windows}
    return [w for w in real_windows if (round(w["t_start"], 3), round(w["t_end"], 3)) in covered]


def _track(directory: Path, subject: str) -> Path:
    path = directory / f"track_{subject}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no track for subject {subject}: expected {path}")
    return path


def run_delta(
    *,
    real_dir: Path | None = None,
    simulate_label_noise: float | None = None,
    simulate_burst_noise: float | None = None,
    seed: int = 0,
    questions_dir: Path = QUESTIONS_DIR,
    oracle_dir: Path = FIXTURES_DIR,
    labelled_only: bool = True,
) -> dict[str, Any]:
    """Compare every dev subject's oracle track with its real track. Pass
    exactly one source: `real_dir` (a directory of track_<subject>.jsonl
    files), or a dry run against a corrupted oracle -- `simulate_label_noise`
    flips individual windows, `simulate_burst_noise` misclassifies whole
    bursts, the correlated error a real classifier makes."""
    sources = [s for s in (real_dir, simulate_label_noise, simulate_burst_noise) if s is not None]
    if len(sources) != 1:
        raise ValueError("pass exactly one of real_dir, simulate_label_noise or simulate_burst_noise")

    subjects = dev_subjects(questions_dir)
    rows: list[dict[str, Any]] = []
    withheld: dict[str, dict[str, int]] = {}
    unlabelled: dict[str, int] = {}
    for index, subject in enumerate(subjects):
        questions = read_question_set(questions_dir / f"{subject}.json")["questions"]
        oracle_windows = load_track(_track(oracle_dir, subject))
        if real_dir is not None:
            real_windows = load_track(_track(real_dir, subject))
        elif simulate_label_noise is not None:
            real_windows = perturb_labels(oracle_windows, simulate_label_noise, seed + index)
        else:
            real_windows = perturb_bursts(oracle_windows, simulate_burst_noise, seed + index)
        unlabelled[subject] = 0
        if labelled_only:
            kept = labelled_windows(real_windows, oracle_windows)
            unlabelled[subject] = len(real_windows) - len(kept)
            real_windows = kept
        subject_rows, withheld[subject] = compare_subject(subject, questions, oracle_windows, real_windows)
        rows.extend(subject_rows)

    if real_dir is not None:
        source = str(real_dir)
    elif simulate_label_noise is not None:
        source = f"SIMULATED: the oracle track with {simulate_label_noise:.0%} of windows mislabelled (seed {seed})"
    else:
        source = f"SIMULATED: the oracle track with {simulate_burst_noise:.0%} of bursts mislabelled (seed {seed})"
    return {
        "real_source": source,
        "oracle_source": str(oracle_dir),
        "questions_source": str(questions_dir),
        "subjects": subjects,
        "by_question_type": summarize(rows),
        "withheld_by_validator": withheld,
        "labelled_only": labelled_only,
        "real_windows_outside_labelled_time": unlabelled,
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    dropped = sum(report["real_windows_outside_labelled_time"].values())
    compared_over = (
        f"labelled time only ({dropped} real-model windows from minutes nobody labelled were set aside)"
        if report["labelled_only"]
        else "every recorded window, including minutes nobody labelled"
    )
    lines = [
        "# Oracle vs real delta",
        "",
        f"- Real track: {report['real_source']}",
        f"- Oracle track: {report['oracle_source']}",
        f"- Questions: {report['questions_source']}",
        f"- Subjects: {', '.join(report['subjects'])}",
        f"- Compared over: {compared_over}",
        "",
        '"Right" means correct, and grounded wherever the gold cites evidence (PRD 7.3.4).',
        "",
        "| Question type | n | Right (oracle) | Right (real) | Lost: recognition (A) | Lost: routing (B) | Lost: reasoning (B) | Masked bugs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in report["by_question_type"].items():
        label = f"**{name}**" if name == "overall" else name
        losses = row["losses"]
        lines.append(
            f"| {label} | {row['n']} | {row['oracle_right']} | {row['real_right']} | "
            f"{losses['recognition']} | {losses['routing']} | {losses['reasoning']} | {row['masked_bugs']} |"
        )

    withheld_oracle = sum(w["oracle"] for w in report["withheld_by_validator"].values())
    withheld_real = sum(w["real"] for w in report["withheld_by_validator"].values())
    lines += [
        "",
        f"Answers withheld by the grounding validator: {withheld_oracle} on the oracle track, {withheld_real} on the real track.",
        "",
    ]

    def fix_list(title: str, layers: tuple[str, ...]) -> None:
        items = [r for r in report["rows"] if r["layer"] in layers]
        lines.extend([f"## Fix list - {title}", ""])
        if not items:
            lines.extend(["None.", ""])
            return
        for r in items:
            tag = r["layer"] if r["is_loss"] else f"{r['layer']}, masked"
            lines.append(
                f"- `{r['question_id']}` ({r['question_type']}, operator `{r['operator']}`, {tag}): "
                f'gold "{r["gold_answer"]}"; oracle "{r["oracle_answer"]}"; real "{r["real_answer"]}"'
            )
        lines.append("")

    fix_list("Member A (recognition)", ("recognition",))
    fix_list("Member B (routing and reasoning)", ("routing", "reasoning"))
    return "\n".join(lines)
