"""Dev-set evaluation harness: answer each dev subject's question set against
that subject's own track, then score everything together. Measures the
Phase 2 exit criteria (docs/TASKS.md) and backs scripts/run_dev_eval.py.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Sequence

from ats.aggregate import load_track
from ats.answer import answer_all
from ats.eval import evaluate
from ats.serialize import read_question_set

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_DIR = REPO_ROOT / "data" / "questions_dev"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def _top(probs: Sequence[float]) -> int:
    return max(range(len(probs)), key=probs.__getitem__)


def perturb_labels(windows: Sequence[dict[str, Any]], p: float, seed: int) -> list[dict[str, Any]]:
    """Give a fraction `p` of windows a wrong predicted class by swapping the
    top probability onto a random other class -- the same corruption as
    ats.oracle's --label-noise, applied to a track that already exists."""
    rng = random.Random(seed)
    perturbed: list[dict[str, Any]] = []
    for window in windows:
        window = copy.deepcopy(window)
        if rng.random() < p:
            probs = window["probs"]
            top = _top(probs)
            other = rng.choice([i for i in range(len(probs)) if i != top])
            probs[top], probs[other] = probs[other], probs[top]
        perturbed.append(window)
    return perturbed


def _bursts(windows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group windows that overlap or touch; a new group starts only after
    dead time. Deliberately independent of ats.aggregate's own chunking."""
    groups: list[list[dict[str, Any]]] = []
    for window in sorted(windows, key=lambda w: w["t_start"]):
        if groups and window["t_start"] <= groups[-1][-1]["t_end"]:
            groups[-1].append(window)
        else:
            groups.append([window])
    return groups


def perturb_bursts(windows: Sequence[dict[str, Any]], p: float, seed: int) -> list[dict[str, Any]]:
    """Misclassify a fraction `p` of whole bursts, every window consistently
    -- the correlated error a real classifier makes when it gets a whole
    minute wrong, which independent per-window flips do not simulate."""
    rng = random.Random(seed)
    perturbed: list[dict[str, Any]] = []
    for group in _bursts(windows):
        group = copy.deepcopy(group)
        if rng.random() < p:
            n_classes = len(group[0]["probs"])
            totals = [sum(w["probs"][i] for w in group) for i in range(n_classes)]
            wrong = rng.choice([i for i in range(n_classes) if i != _top(totals)])
            for window in group:
                probs = window["probs"]
                top = _top(probs)
                probs[top], probs[wrong] = probs[wrong], probs[top]
        perturbed.extend(group)
    return perturbed


def dev_subjects(questions_dir: Path = QUESTIONS_DIR) -> list[str]:
    return sorted(path.stem for path in questions_dir.glob("*.json"))


def run_dev_eval(
    *,
    label_noise: float = 0.0,
    burst_noise: float = 0.0,
    seed: int = 0,
    questions_dir: Path = QUESTIONS_DIR,
    fixtures_dir: Path = FIXTURES_DIR,
) -> dict[str, Any]:
    subjects = dev_subjects(questions_dir)
    all_answers: list[dict[str, Any]] = []
    all_questions: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    for index, subject in enumerate(subjects):
        questions = read_question_set(questions_dir / f"{subject}.json")["questions"]
        windows = load_track(fixtures_dir / f"track_{subject}.jsonl")
        if label_noise:
            windows = perturb_labels(windows, label_noise, seed + index)
        if burst_noise:
            windows = perturb_bursts(windows, burst_noise, seed + index)
        answers, subject_rejections = answer_all(questions, windows)
        all_answers.extend(answers)
        all_questions.extend(questions)
        rejections.extend(subject_rejections)

    report = evaluate({"answers": all_answers}, {"questions": all_questions})
    report.update(
        {
            "subjects": subjects,
            "label_noise": label_noise,
            "burst_noise": burst_noise,
            "seed": seed,
            "n_rejected_by_validator": len(rejections),
            "rejections": rejections,
            "n_abstained": sum(1 for a in all_answers if a["answer"].strip().upper() == "N/A"),
            "n_empty_answers": sum(1 for a in all_answers if not a["answer"].strip()),
        }
    )
    return report
