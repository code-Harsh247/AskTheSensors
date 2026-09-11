"""Figure 5 (PRD Sec 7.4.5, docs/TASKS.md task 5A.4): accuracy vs. signal
degradation for the `full` config, one curve per degradation axis.

Reads results/robustness.csv (scripts/sweep.py). Writes:
    results/fig5_robustness.png
    results/fig5_robustness.csv   (the table view, same rows as robustness.csv)
    results/fig5_robustness.md    (caption: axes, levels, accuracy definition)

Usage: python scripts/make_fig5.py [--robustness results/robustness.csv]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _figure_style as style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ats.eval import metrics  # noqa: E402

DEFAULT_ROBUSTNESS = REPO_ROOT / "results" / "robustness.csv"
OUT = REPO_ROOT / "results" / "fig5_robustness"

AXIS_LABELS = {
    "dropout": "Fraction of samples dropped",
    "noise": "Additive noise SNR (dB) -- lower is noisier",
    "decimate": "Decimation factor (keep every Nth sample)",
}
COLORS = [style.ORACLE, style.REAL_MODEL, style.MUTED]


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["level"] = float(row["level"])
        row["accuracy"] = float(row["accuracy"])
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robustness", default=str(DEFAULT_ROBUSTNESS))
    args = parser.parse_args(argv)

    rows = _read_rows(Path(args.robustness))
    baseline = next((r for r in rows if r["axis"] == "none"), None)
    axes = sorted({r["axis"] for r in rows if r["axis"] != "none"})
    if not axes:
        raise ValueError(f"{args.robustness} has no degraded rows -- run scripts/sweep.py first")

    style.apply()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    style.recede(ax)

    # Each axis has its own natural "no degradation" point: dropout=0.0,
    # decimate=1 (keep every sample); noise has no finite "clean" SNR, so
    # its curve is plotted on its own without the shared baseline spliced in.
    NO_DEGRADATION_LEVEL = {"dropout": 0.0, "decimate": 1.0}

    for i, axis in enumerate(axes):
        axis_rows = sorted((r for r in rows if r["axis"] == axis), key=lambda r: r["level"])
        levels = [r["level"] for r in axis_rows]
        accuracies = [r["accuracy"] for r in axis_rows]
        if baseline is not None and axis in NO_DEGRADATION_LEVEL:
            levels = [NO_DEGRADATION_LEVEL[axis]] + levels
            accuracies = [baseline["accuracy"]] + accuracies
        ax.plot(levels, accuracies, marker="o", markersize=style.MARKER_SIZE, linewidth=style.LINE_WIDTH, color=COLORS[i % len(COLORS)], label=axis)

    # Each axis's level uses different, incompatible units (a dropout
    # fraction, an SNR in dB, a decimation factor) -- a single shared x-axis
    # only makes sense plotted alone or read per-curve from the legend.
    xlabel = AXIS_LABELS[axes[0]] if len(axes) == 1 else "Degradation level (units vary by axis -- see legend)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (overall macro, ats.eval.evaluate)")
    ax.set_title("Figure 5: Accuracy vs. signal degradation (full config)", loc="left", fontsize=10)
    ax.legend(loc="best", handlelength=1.2)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=style.DPI)
    plt.close(fig)

    shutil.copy(args.robustness, OUT.with_suffix(".csv"))

    caption = [
        "**Figure 5: Accuracy vs. signal degradation, `full` config.** "
        "Accuracy is `ats.eval.evaluate`'s overall macro accuracy over the frozen "
        f"`data/questions_dev_v2` question set -- the same pre-registered rules as Figures 1 and 4 "
        f"(IoU >= {metrics.HEADLINE_IOU_THRESHOLD:g} for grounding, "
        f"max({metrics.DURATION_ABS_TOL_S:g}s, {metrics.RELATIVE_DURATION_TOL:.0%}) for duration, "
        f"+/-{metrics.COUNT_ABS_TOL:g} bout for count). "
        "Degradation is applied to the raw resampled signal before windowing "
        "(`ats/degrade.py`), so it changes both the model's input and every "
        "`feature_summary` value an explanation would cite, not just the model's tensor. "
        "Axes plotted: " + ", ".join(f"**{a}** ({AXIS_LABELS[a].lower()})" for a in axes) + ". "
        "The \"none\" point (no degradation) anchors every curve at its clean baseline.",
        "",
        "Produced by `scripts/make_fig5.py`; numbers in `results/fig5_robustness.csv`.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(caption) + "\n", encoding="utf-8")

    print(f"-> {OUT.with_suffix('.png').relative_to(REPO_ROOT)} (+ .csv, .md)")


if __name__ == "__main__":
    main()
