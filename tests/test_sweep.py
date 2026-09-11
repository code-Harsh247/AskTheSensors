"""scripts/sweep.py: only its pure helper is unit-tested here. The rest
(run_pareto/run_robustness) is thin orchestration over ats.recognize,
ats.answer, ats.eval, and ats.degrade -- each already tested on its own;
re-mocking the whole pipeline here would test the mocks, not the code."""

from __future__ import annotations

import csv

import pytest

import json

from scripts.sweep import COST_KEYS, _write_csv, costs_from_report, refresh_costs


def _cost_report(results_dir, config, latency):
    report = {key: 1.0 for key in COST_KEYS} | {"latency_p50_ms": latency, "config_id": config}
    (results_dir / f"cost_report_{config}.json").write_text(json.dumps(report), encoding="utf-8")


def test_cost_columns_come_from_the_standalone_cost_report(tmp_path):
    _cost_report(tmp_path, "quant8", 0.353)
    assert costs_from_report("quant8", tmp_path)["latency_p50_ms"] == pytest.approx(0.353)
    with pytest.raises(FileNotFoundError):
        costs_from_report("pruned30", tmp_path)


def test_refresh_costs_keeps_accuracy_and_replaces_every_cost_column(tmp_path):
    _cost_report(tmp_path, "full", 0.397)
    _cost_report(tmp_path, "quant8", 0.353)
    stale = [
        {"config_id": "full", "accuracy": 0.39, **{key: 999.0 for key in COST_KEYS}},
        {"config_id": "quant8", "accuracy": 0.45, **{key: 999.0 for key in COST_KEYS}},
    ]
    _write_csv(stale, tmp_path / "pareto.csv")

    rows = refresh_costs(tmp_path / "pareto.csv", tmp_path)

    assert [r["accuracy"] for r in rows] == [0.39, 0.45]
    assert [r["latency_p50_ms"] for r in rows] == [0.397, 0.353]
    assert all(r[key] != 999.0 for r in rows for key in COST_KEYS)


def test_write_csv_round_trips_rows(tmp_path):
    rows = [
        {"config_id": "full", "accuracy": 0.39, "disk_mb": 0.047},
        {"config_id": "quant8", "accuracy": 0.37, "disk_mb": 0.034},
    ]
    out_path = tmp_path / "pareto.csv"

    _write_csv(rows, out_path)

    with out_path.open(newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 2
    assert written[0]["config_id"] == "full"
    assert float(written[1]["disk_mb"]) == pytest.approx(0.034)


def test_write_csv_raises_on_no_rows(tmp_path):
    with pytest.raises(ValueError):
        _write_csv([], tmp_path / "empty.csv")
