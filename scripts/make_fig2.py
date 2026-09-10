"""Figure 2 (PRD Sec 7.4.2, docs/TASKS.md 2A.5): the recognition backbone's
confusion matrix over the 7 canonical classes, plus a per-class
precision/recall/F1 table.

Reads the results `scripts/train_cnn.py` produces on Kaggle
(`phase2_results.json`), downloaded and placed at `models/full/` locally
(see that script's docstring). Writes:
    results/fig2_confusion_matrix.png
    results/fig2_per_class_prf.csv          (small summary, committed)

Usage: python scripts/make_fig2.py [--results models/full/phase2_results.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ats.contracts import CANONICAL_CLASSES  # noqa: E402

DEFAULT_RESULTS = REPO_ROOT / "models" / "full" / "phase2_results.json"
FIG_OUT = REPO_ROOT / "results" / "fig2_confusion_matrix.png"
TABLE_OUT = REPO_ROOT / "results" / "fig2_per_class_prf.csv"


def plot_confusion_matrix(matrix: list[list[int]], out_path: Path) -> None:
    arr = np.asarray(matrix, dtype=int)
    row_sums = arr.sum(axis=1, keepdims=True)
    normalized = np.divide(arr, row_sums, out=np.zeros_like(arr, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CANONICAL_CLASSES)))
    ax.set_yticks(range(len(CANONICAL_CLASSES)))
    ax.set_xticklabels(CANONICAL_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CANONICAL_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Figure 2: Activity confusion matrix (window-level, val split)\nCell = fraction of true row's windows")

    for i in range(len(CANONICAL_CLASSES)):
        for j in range(len(CANONICAL_CLASSES)):
            ax.text(
                j, i, str(arr[i, j]),
                ha="center", va="center",
                color="white" if normalized[i, j] > 0.5 else "black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, label="fraction of true-row windows")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_per_class_table(per_class: dict, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for cls in CANONICAL_CLASSES:
            row = per_class[cls]
            writer.writerow([cls, row["precision"], row["recall"], row["f1"], row["support"]])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    args = parser.parse_args(argv)

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    cnn = results["cnn"]
    plot_confusion_matrix(cnn["confusion_matrix"], FIG_OUT)
    write_per_class_table(cnn["per_class"], TABLE_OUT)
    print(f"wrote {FIG_OUT.relative_to(REPO_ROOT)} and {TABLE_OUT.relative_to(REPO_ROOT)}")
    print(f"macro-F1 (val): {cnn['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
