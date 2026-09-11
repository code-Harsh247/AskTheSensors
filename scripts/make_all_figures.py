"""Regenerate all five report figures with zero manual steps (docs/TASKS.md
Phase 5 exit criterion). Each figure script already has sensible defaults
(reads from tests/fixtures, data/questions_dev_v2, results/*.csv); this just
calls them in order from a clean checkout.

Requires the inputs each figure needs to already exist:
    Figures 1, 3: tests/fixtures/, tests/fixtures/real_model_tracks/, data/questions_dev_v2/
    Figure 2: models/full/phase2_results.json (scripts/train_cnn.py)
    Figures 4, 5: results/pareto.csv, results/robustness.csv (scripts/sweep.py)

Usage: python scripts/make_all_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_fig1  # noqa: E402
import make_fig2  # noqa: E402
import make_fig3  # noqa: E402
import make_fig4  # noqa: E402
import make_fig5  # noqa: E402


TAKES_ARGV = {"Figure 2", "Figure 4", "Figure 5"}  # make_fig1/make_fig3's main() takes no argv param


def main() -> None:
    for name, module in (
        ("Figure 1", make_fig1),
        ("Figure 2", make_fig2),
        ("Figure 3", make_fig3),
        ("Figure 4", make_fig4),
        ("Figure 5", make_fig5),
    ):
        print(f"=== {name} ({module.__name__}) ===")
        module.main([]) if name in TAKES_ARGV else module.main()
    print("\nAll five figures regenerated.")


if __name__ == "__main__":
    main()
