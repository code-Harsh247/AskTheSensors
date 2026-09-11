"""Phase 4 exit criterion (docs/TASKS.md): no ungrounded answer reaches the
output. Every dev question is answered against every track available --
synthetic oracle fixtures, both real subjects on their oracle and real-model
tracks, and synthetic tracks with whole bursts mislabelled -- and every
emitted answer must pass the grounding validator. Any tier-3 or tier-4 answer
that is not an explicit abstention must cite evidence that exists in the
timeline."""

from functools import lru_cache

import pytest

from ats.aggregate import build_timeline, load_track
from ats.answer import answer_all
from ats.eval.dev import FIXTURES_DIR, QUESTIONS_DIR, REPO_ROOT, dev_subjects, perturb_bursts
from ats.routing import route
from ats.serialize import read_question_set
from ats.validator import grounding_problems, is_abstention

QUESTIONS_V2 = REPO_ROOT / "data" / "questions_dev_v2"
REAL_MODEL_TRACKS = FIXTURES_DIR / "real_model_tracks"

CASES = [
    *[(QUESTIONS_DIR, s, FIXTURES_DIR, "oracle") for s in dev_subjects(QUESTIONS_DIR)],
    *[(QUESTIONS_DIR, s, FIXTURES_DIR, "burst-noise") for s in dev_subjects(QUESTIONS_DIR)],
    *[(QUESTIONS_V2, s, FIXTURES_DIR, "oracle") for s in dev_subjects(QUESTIONS_V2)],
    *[(QUESTIONS_V2, s, REAL_MODEL_TRACKS, "real-model") for s in dev_subjects(QUESTIONS_V2)],
]


@lru_cache(maxsize=None)
def _answered(questions_dir, subject, track_dir, kind):
    questions = read_question_set(questions_dir / f"{subject}.json")["questions"]
    windows = load_track(track_dir / f"track_{subject}.jsonl")
    if kind == "burst-noise":
        windows = perturb_bursts(windows, 0.2, seed=0)
    answers, rejections = answer_all(questions, windows)
    return questions, answers, rejections, build_timeline(windows)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[1]}-{c[3]}")
def test_every_emitted_answer_is_grounded(case):
    questions, answers, _, timeline = _answered(*case)
    for question, answer in zip(questions, answers):
        assert grounding_problems(answer, route(question["text"]), timeline) == [], question["question_id"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[1]}-{c[3]}")
def test_tier_3_and_4_claims_cite_evidence(case):
    _, answers, _, _ = _answered(*case)
    claims = [a for a in answers if a["tier_inferred"] >= 3 and not is_abstention(a)]
    assert claims, "the case should exercise tier-3/4 claims"
    for answer in claims:
        assert answer["cited_intervals"], answer["question_id"]
        assert answer["evidence"]["timestamps"] != "N/A", answer["question_id"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[1]}-{c[3]}")
def test_operators_never_needed_to_be_overruled(case):
    """Stronger than the criterion: the operators' own output was grounded,
    so the validator never had to replace an answer with an abstention."""
    _, _, rejections, _ = _answered(*case)
    assert rejections == []
