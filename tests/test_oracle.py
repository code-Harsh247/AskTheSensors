"""ats/ingest.py + ats/oracle.py against a small, fully-controlled fixture in
the exact ExtraSensory archive format (docs/CITATIONS.md#extrasensory-raw-file-layout),
so correctness here does not depend on the live dataset having been fetched.
"""

from __future__ import annotations

import csv
import gzip
import io
import zipfile
from pathlib import Path

import pytest

from ats.aggregate import build_timeline, load_track
from ats.contracts import CANONICAL_CLASSES, validate_window_track
from ats.ingest import LABEL_COLUMNS, load_subject
from ats.oracle import build_track
from ats.resample import globalize_subject
from ats.windowing import HOP_S, WINDOW_LENGTH_S, make_windows

SUBJECT = "TEST-UUID-0000"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _burst_dat(t0: float, n: int, hz: float, x: float, y: float, z: float) -> bytes:
    dt = 1.0 / hz
    lines = [f"{t0 + i * dt:.6f} {x:.6f} {y:.6f} {z:.6f}" for i in range(n)]
    return "\n".join(lines).encode("ascii")


def _write_zip(path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def _labels_csv_gz(rows: list[dict[str, str]]) -> bytes:
    fieldnames = ["timestamp"] + [col for col, _ in LABEL_COLUMNS]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "0") for name in fieldnames})
    return gzip.compress(buf.getvalue().encode("ascii"))


@pytest.fixture
def data_dir(tmp_path):
    meta = tmp_path / "_meta"
    meta.mkdir()

    # Three examples, 60s apart in wall-clock terms (only ~20s of each minute
    # is actually recorded -- real ExtraSensory duty cycle):
    #   ts=1000: WALKING, clean burst
    #   ts=1060: ambiguous (two main-activity columns fire) -> must be skipped
    #   ts=1120: SITTING, clean burst
    rows = [
        {"timestamp": "1000", "original_label:WALKING": "1"},
        {"timestamp": "1060", "original_label:SITTING": "1", "original_label:STANDING_IN_PLACE": "1"},
        {"timestamp": "1120", "original_label:SITTING": "1"},
    ]
    _write_zip(meta / "original_labels.zip", {f"{SUBJECT}.original_labels.csv.gz": _labels_csv_gz(rows)})

    acc_entries = {}
    gyro_entries = {}
    for ts, (x, y, z) in {1000: (0.3, 0.1, 1.0), 1060: (0.0, 0.0, 1.0), 1120: (0.0, 0.0, 1.0)}.items():
        acc_entries[f"raw_acc/{SUBJECT}/{ts}.m_raw_acc.dat"] = _burst_dat(500.0, 800, 40.0, x, y, z)
        gyro_entries[f"proc_gyro/{SUBJECT}/{ts}.m_proc_gyro.dat"] = _burst_dat(500.0, 800, 40.0, 0.01, 0.0, 0.0)
    _write_zip(meta / "raw_acc.zip", acc_entries)
    _write_zip(meta / "proc_gyro.zip", gyro_entries)

    return tmp_path


def test_load_subject_maps_labels_and_flags_ambiguous_as_none(data_dir):
    subject = load_subject(data_dir, SUBJECT)
    by_ts = {b.example_ts: b.activity for b in subject.bursts}
    assert by_ts[1000.0] == "WALKING"
    assert by_ts[1060.0] is None  # two main-activity columns fired: ambiguous
    assert by_ts[1120.0] == "SITTING"


def test_accelerometer_units_converted_to_ms2(data_dir):
    subject = load_subject(data_dir, SUBJECT)
    burst = next(b for b in subject.bursts if b.example_ts == 1000.0)
    # Raw value was 1.0g on z; ingest.py converts to m/s^2.
    assert burst.acc[0][3] == pytest.approx(9.80665)


def test_globalize_anchors_bursts_with_real_gaps_between_them(data_dir):
    subject = load_subject(data_dir, SUBJECT)
    globalized = globalize_subject(subject)
    # First burst anchored at t=0; second is 60s later in wall-clock terms.
    assert globalized.label_spans[0][1] == 0.0
    assert globalized.label_spans[1][1] == pytest.approx(60.0)
    assert globalized.label_spans[2][1] == pytest.approx(120.0)
    # Within a burst, spacing follows the device clock (40Hz -> ~20s span).
    span0 = globalized.label_spans[0]
    assert span0[2] - span0[1] == pytest.approx(19.975, abs=1e-3)


def test_make_windows_never_spans_the_dead_time_between_bursts(data_dir):
    # ats/windowing.py:make_windows generates windows per burst span, not by
    # blindly tiling [0, t_end] -- a subject's bursts are scattered across a
    # multi-day span that is mostly dead time (only ~20s of every ~60s is
    # ever recorded), so every window it emits must fall entirely within one
    # burst's span.
    subject = load_subject(data_dir, SUBJECT)
    globalized = globalize_subject(subject)
    windows = make_windows(globalized, window_s=WINDOW_LENGTH_S, hop_s=HOP_S)
    assert windows, "fixture should produce at least one window"

    burst_spans = [(0.0, 19.975), (60.0, 79.975), (120.0, 139.975)]
    for w in windows:
        assert any(
            start <= w.t_start and w.t_end <= end for start, end in burst_spans
        ), (w.t_start, w.t_end)

    # build_track (ats/oracle.py) then additionally drops the ambiguous
    # (1060) burst's windows for lacking a usable single label.
    for entry in build_track(SUBJECT, data_dir):
        assert any(
            start <= entry["t_start"] and entry["t_end"] <= end for start, end in burst_spans
        ), (entry["t_start"], entry["t_end"])


def test_build_track_emits_schema_valid_entries_only_for_labeled_bursts(data_dir):
    entries = build_track(SUBJECT, data_dir)
    assert entries, "expected at least one window from the two clean bursts"
    for entry in entries:
        validate_window_track(entry)
        assert entry["model_id"] == "oracle"

    # No window should ever be attributed to the ambiguous (1060) burst.
    ambiguous_windows = [e for e in entries if 60.0 <= e["t_start"] < 80.0]
    assert ambiguous_windows == []

    # One-hot by default (no --soften-confidence).
    for entry in entries:
        assert sorted(entry["probs"], reverse=True)[0] == 1.0


def test_build_track_soften_confidence_produces_non_one_hot_probs(data_dir):
    entries = build_track(SUBJECT, data_dir, soften_confidence=True, seed=1)
    assert any(max(e["probs"]) < 1.0 for e in entries)
    for entry in entries:
        assert abs(sum(entry["probs"]) - 1.0) < 1e-9


def test_build_track_label_noise_flips_every_label_at_probability_one(data_dir):
    clean = build_track(SUBJECT, data_dir, seed=0)
    noisy = build_track(SUBJECT, data_dir, label_noise=1.0, seed=0)
    assert len(clean) == len(noisy)
    for c, n in zip(clean, noisy):
        clean_label = CANONICAL_CLASSES[c["probs"].index(max(c["probs"]))]
        noisy_label = CANONICAL_CLASSES[n["probs"].index(max(n["probs"]))]
        assert noisy_label != clean_label


def test_build_track_drop_windows_reduces_output_count(data_dir):
    full = build_track(SUBJECT, data_dir, seed=0)
    dropped = build_track(SUBJECT, data_dir, drop_windows=1.0, seed=0)
    assert dropped == []
    assert len(full) > 0


def test_committed_real_sample_track_is_schema_valid_and_aggregates():
    """The real-subject excerpt committed for Member B (docs/TASKS.md
    "Artifacts crossing the boundary": one committed sample track), generated
    by scripts/make_sample_track.py, round-trips through B's aggregation
    layer -- proof this isn't just schema-shaped, it's usable."""
    path = FIXTURES_DIR / "track_subj_real_00EABED2.jsonl"
    windows = load_track(path)
    assert len(windows) > 0
    for entry in windows:
        validate_window_track(entry)

    timeline = build_timeline(windows)
    assert timeline.intervals, "expected at least one activity interval"
    seen_activities = timeline.activities_present()
    assert seen_activities <= set(CANONICAL_CLASSES)
    assert len(seen_activities) > 1, "sample track should span more than one real activity"
