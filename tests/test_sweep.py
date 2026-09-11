"""scripts/sweep.py: only its pure helper is unit-tested here. The rest
(run_pareto/run_robustness) is thin orchestration over ats.recognize,
ats.answer, ats.eval, and ats.degrade -- each already tested on its own;
re-mocking the whole pipeline here would test the mocks, not the code."""

from __future__ import annotations

import csv

import pytest

from scripts.sweep import _write_csv


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
