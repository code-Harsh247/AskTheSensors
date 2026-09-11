"""The evaluation-time entry point: a raw recording and a question set in,
PRD §5 answers out, with the recognition model in between. An untrained
ActivityCNN stands in for the trained weights, as in tests/test_recognize.py,
so these tests check the plumbing, not recognition accuracy."""

import json

import pytest
import torch

from ats import answer
from ats.contracts import validate_window_track
from ats.ingest import Burst, Subject, load_subject
from ats.model import ActivityCNN
from ats.serialize import read_answers_jsonl

QUESTIONS = {
    "questions": [
        {"question_id": "q1", "text": "What activity is the user performing?"},
        {"question_id": "q2", "text": "How long was the user sitting in total?"},
        {"question_id": "q3", "text": "Did the user begin walking at any point, and if so, when?"},
    ]
}


def _unlabelled_burst(ts, n=800, hz=40.0):
    dt = 1.0 / hz
    return Burst(
        example_ts=ts,
        acc=tuple((i * dt, 0.0, 0.0, 9.8) for i in range(n)),
        gyro=tuple((i * dt, 0.0, 0.0, 0.0) for i in range(n)),
        activity=None,
    )


@pytest.fixture
def recording(tmp_path, monkeypatch):
    subject = Subject("subj-x", tuple(_unlabelled_burst(1000.0 + 60.0 * k) for k in range(5)))
    monkeypatch.setattr("ats.recognize.load_subject", lambda data_dir, subject_id: subject)
    model_path = tmp_path / "model.pt"
    torch.manual_seed(0)
    torch.save(ActivityCNN().state_dict(), model_path)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    return tmp_path, model_path, questions


def test_a_recording_and_questions_go_in_and_valid_answers_come_out(recording):
    tmp, model_path, questions = recording
    out, track = tmp / "answers.jsonl", tmp / "track.jsonl"

    answer.main(
        [
            "--recording", str(tmp), "--subject", "subj-x", "--model", str(model_path),
            "--questions", str(questions), "--out", str(out), "--format", "jsonl",
            "--save-track", str(track),
        ]
    )

    answers = read_answers_jsonl(out)  # validates every answer against the schema
    assert [a["question_id"] for a in answers] == ["q1", "q2", "q3"]
    windows = [json.loads(line) for line in track.read_text(encoding="utf-8").splitlines()]
    assert windows
    for window in windows:
        validate_window_track(window)


@pytest.mark.parametrize(
    "extra",
    [
        [],  # neither a recording nor a track
        ["--recording", "somewhere"],  # a recording without its subject
        ["--recording", "somewhere", "--subject", "s", "--track", "t.jsonl"],  # both
    ],
    ids=["nothing", "no-subject", "both"],
)
def test_the_input_must_be_exactly_one_recording_or_track(extra, tmp_path):
    with pytest.raises(SystemExit):
        answer.main(["--questions", "q.json", "--out", str(tmp_path / "a.txt"), *extra])


@pytest.mark.xfail(
    strict=True,
    raises=FileNotFoundError,
    reason="docs/bug.md issue 2: ats.ingest.load_subject requires the labels archive, "
    "which an evaluation-time recording will not have",
)
def test_an_unlabelled_recording_can_be_loaded(tmp_path):
    """A minimal real-format recording with no labels archive. When Member A's
    fix lands this passes, strict xfail turns that into a failure, and the
    marker should be removed in the same change."""
    subject = "UNLABELLED-SUBJECT"
    for channel, suffix in (("raw_acc", ".m_raw_acc.dat"), ("proc_gyro", ".m_proc_gyro.dat")):
        folder = tmp_path / "_meta" / channel / subject
        folder.mkdir(parents=True)
        rows = "\n".join(f"{i / 40:.3f} 0.0 0.0 1.0" for i in range(800))
        (folder / f"1000{suffix}").write_text(rows, encoding="ascii")

    loaded = load_subject(tmp_path, subject)
    assert len(loaded.bursts) == 1
    assert loaded.bursts[0].activity is None
