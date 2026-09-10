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
            top = max(range(len(probs)), key=probs.__getitem__)
            other = rng.choice([i for i in range(len(probs)) if i != top])
            probs[top], probs[other] = probs[other], probs[top]
        perturbed.append(window)
    return perturbed


def dev_subjects(questions_dir: Path = QUESTIONS_DIR) -> list[str]:
    return sorted(path.stem for path in questions_dir.glob("*.json"))


def run_dev_eval(
    *,
    label_noise: float = 0.0,
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
        answers, subject_rejections = answer_all(questions, windows)
        all_answers.extend(answers)
        all_questions.extend(questions)
        rejections.extend(subject_rejections)

    report = evaluate({"answers": all_answers}, {"questions": all_questions})
    report.update(
        {
            "subjects": subjects,
            "label_noise": label_noise,
            "seed": seed,
            "n_rejected_by_validator": len(rejections),
            "rejections": rejections,
            "n_abstained": sum(1 for a in all_answers if a["answer"].strip().upper() == "N/A"),
            "n_empty_answers": sum(1 for a in all_answers if not a["answer"].strip()),
        }
    )
    return report
