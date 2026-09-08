"""ats.eval exposes `evaluate(pred, gold) -> dict` as an importable function
(TASKS.md Phase 0, task 0.6) so the Phase 5 sweep can call it directly per
config/degradation point instead of shelling out. The full PRD §7.3 metric
library lands in ats/eval/metrics.py (Phase 1, task 1B.3).
"""

from __future__ import annotations

from typing import Any


def evaluate(pred: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Score predicted answers against gold. Stub for Phase 0; the PRD §7.3
    metric suite (accuracy, macro-F1, IoU, grounded accuracy, ...) is built
    out in ats/eval/metrics.py during Phase 1.
    """
    raise NotImplementedError(
        "ats.eval.evaluate is a Phase 0 stub. The metric library lands in Phase 1 (docs/TASKS.md task 1B.3)."
    )
