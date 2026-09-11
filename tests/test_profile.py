"""ats/profile.py (docs/TASKS.md task 4A.2): the cost-measurement harness.
An untrained ActivityCNN stands in for trained weights (same convention as
tests/test_recognize.py) -- correctness of the model itself is out of
scope here; this only checks the harness measures and reports correctly."""

from __future__ import annotations

import json

import pytest
import torch

from ats import profile
from ats.contracts import validate_cost_report
from ats.model import ActivityCNN, count_parameters


def test_time_calls_computes_hand_worked_percentiles(monkeypatch):
    """Five calls taking 10/20/30/40/50 ms: nearest-rank p50 is index
    round(0.5*4)=2 -> 30ms, p95 is index round(0.95*4)=4 -> 50ms. Fed via a
    scripted time.perf_counter() so the durations are exact, not measured."""
    durations_ms = [10, 20, 30, 40, 50]
    timestamps = []
    cum = 0.0
    for d in durations_ms:
        timestamps.append(cum)
        cum += d / 1000.0
        timestamps.append(cum)
    it = iter(timestamps)
    monkeypatch.setattr(profile.time, "perf_counter", lambda: next(it))

    p50, p95 = profile.time_calls(lambda: None, n=5)

    assert p50 == pytest.approx(30.0)
    assert p95 == pytest.approx(50.0)


def test_peak_rss_mb_is_positive():
    assert profile.peak_rss_mb() > 0.0


def test_energy_per_query_is_a_hand_worked_value(monkeypatch):
    """TDP x utilization fraction x time, worked by hand: with 8 logical
    cores, cpu_pct=400 (4 cores busy) means a 0.5 utilization fraction;
    45W x 0.5 x 0.010s = 0.225 J."""
    monkeypatch.setattr(profile.psutil, "cpu_count", lambda logical=True: 8)
    assert profile.energy_per_query_j(cpu_pct=400.0, latency_ms=10.0) == pytest.approx(0.225)


def test_energy_per_query_scales_with_latency(monkeypatch):
    monkeypatch.setattr(profile.psutil, "cpu_count", lambda logical=True: 8)
    short = profile.energy_per_query_j(cpu_pct=100.0, latency_ms=1.0)
    long = profile.energy_per_query_j(cpu_pct=100.0, latency_ms=10.0)
    assert long == pytest.approx(short * 10)


def test_profile_recognition_emits_a_schema_valid_report(tmp_path):
    model = ActivityCNN()
    model_path = tmp_path / "activity_cnn.pt"
    torch.save(model.state_dict(), model_path)

    report = profile.profile_recognition(model_path, n_queries=5)
    report["config_id"] = "full"
    report["target_device"] = profile.TARGET_DEVICE

    validate_cost_report(report)
    assert report["params"] == count_parameters(ActivityCNN())
    assert report["disk_mb"] > 0.0
    assert report["latency_p50_ms"] >= 0.0
    assert report["latency_p95_ms"] >= report["latency_p50_ms"]
    assert 0.0 <= report["cpu_pct"]
    assert report["energy_estimate_j"] >= 0.0


def test_main_writes_a_schema_valid_report_to_disk(tmp_path):
    model_path = tmp_path / "activity_cnn.pt"
    torch.save(ActivityCNN().state_dict(), model_path)
    out_path = tmp_path / "cost_report.json"

    profile.main(
        ["--config", "full", "--model", str(model_path), "--n-queries", "3", "--out", str(out_path)]
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    validate_cost_report(written)
    assert written["config_id"] == "full"
    assert written["target_device"] == profile.TARGET_DEVICE


def test_main_raises_a_clear_error_when_the_model_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        profile.main(["--config", "full", "--model", str(tmp_path / "does_not_exist.pt")])
