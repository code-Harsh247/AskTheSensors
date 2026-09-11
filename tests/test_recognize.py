"""ats/recognize.py: the trained-model counterpart to ats/oracle.py (docs/TASKS.md
task 2A.4). Uses an untrained ActivityCNN to verify the plumbing (shapes,
schema validity, graceful handling of an entirely-missing channel) --
correctness of the *model itself* is covered separately by
tests/test_model.py and the real measured results in docs/results_recognition.md.
"""

from __future__ import annotations

import torch

from ats.contracts import validate_window_track
from ats.ingest import Burst, Subject
from ats.model import ActivityCNN
from ats.recognize import build_track


def _burst(ts: float, n: int, hz: float, acc=True, gyro=True) -> Burst:
    dt = 1.0 / hz
    acc_rows = tuple((ts + i * dt, 0.0, 0.0, 9.8) for i in range(n)) if acc else ()
    gyro_rows = tuple((ts + i * dt, 0.0, 0.0, 0.0) for i in range(n)) if gyro else ()
    return Burst(example_ts=ts, acc=acc_rows, gyro=gyro_rows, activity="SITTING")


def test_build_track_emits_schema_valid_entries(tmp_path, monkeypatch):
    subject = Subject(subject_id="test-subj", bursts=(_burst(1000.0, 800, 40.0),))
    monkeypatch.setattr("ats.recognize.load_subject", lambda data_dir, subject_id: subject)

    model = ActivityCNN()
    entries = build_track("test-subj", tmp_path, model)

    assert len(entries) > 0
    for entry in entries:
        validate_window_track(entry)
        assert entry["model_id"] == "full"
        assert abs(sum(entry["probs"]) - 1.0) < 1e-5


def test_build_track_uses_the_given_model_id(tmp_path, monkeypatch):
    subject = Subject(subject_id="test-subj", bursts=(_burst(1000.0, 800, 40.0),))
    monkeypatch.setattr("ats.recognize.load_subject", lambda data_dir, subject_id: subject)

    model = ActivityCNN()
    entries = build_track("test-subj", tmp_path, model, model_id="quant8")
    assert all(e["model_id"] == "quant8" for e in entries)


def test_build_track_skips_windows_with_an_entirely_missing_channel(tmp_path, monkeypatch):
    """A burst with no accelerometer at all (e.g. after ats.ingest drops a
    corrupted-timestamp channel) has nothing to gap-fill -- those windows
    must be skipped, not crash the whole subject."""
    good_burst = _burst(1000.0, 800, 40.0, acc=True, gyro=True)
    no_acc_burst = _burst(1100.0, 800, 40.0, acc=False, gyro=True)
    subject = Subject(subject_id="test-subj", bursts=(good_burst, no_acc_burst))
    monkeypatch.setattr("ats.recognize.load_subject", lambda data_dir, subject_id: subject)

    model = ActivityCNN()
    entries = build_track("test-subj", tmp_path, model)

    # Only the good burst's windows should appear; none should span the
    # no-acc burst's time range [100.0, ~119.975) (bursts are 100s apart
    # here: example_ts 1000 vs 1100, i.e. offsets 0 and 100).
    assert entries
    assert all(e["t_end"] <= 20.0 for e in entries)


def test_build_track_raises_for_a_subject_with_no_usable_bursts_at_all(tmp_path, monkeypatch):
    """ats.ingest.load_subject already excludes bursts with both channels
    empty, so a subject that reaches here with bursts=() has nothing to
    build a track from at all -- ats.resample.globalize_subject raises
    rather than silently returning an empty track."""
    import pytest

    subject = Subject(subject_id="test-subj", bursts=())
    monkeypatch.setattr("ats.recognize.load_subject", lambda data_dir, subject_id: subject)

    with pytest.raises(ValueError):
        build_track("test-subj", tmp_path, ActivityCNN())
