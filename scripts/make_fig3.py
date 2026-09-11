"""Figure 3 (PRD 7.4.3; docs/TASKS.md 5B.3): accuracy versus strictness.
Left, the fraction of cited intervals accepted as the IoU threshold rises from
0.1 to 0.9; right, the fraction of duration answers accepted as the relative
tolerance widens. The trained model against oracle labels, with the
pre-registered thresholds marked. Writes:
    results/fig3_accuracy_vs_strictness.png
    results/fig3_accuracy_vs_strictness.csv   (the table view)
    results/fig3_accuracy_vs_strictness.md    (caption)

Usage: python scripts/make_fig3.py [--questions-dir data/questions_dev_v2]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _figure_style as style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ats.eval import metrics  # noqa: E402
from ats.eval.curves import answer_sets, iou_acceptance, tolerance_acceptance  # noqa: E402
from ats.eval.dev import FIXTURES_DIR, dev_subjects  # noqa: E402

OUT = REPO_ROOT / "results" / "fig3_accuracy_vs_strictness"
IOU_THRESHOLDS = [round(0.1 + 0.05 * i, 2) for i in range(17)]
TOLERANCES = [round(0.01 * i, 2) for i in range(101)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Figure 3.")
    parser.add_argument("--questions-dir", type=Path, default=REPO_ROOT / "data" / "questions_dev_v2")
    parser.add_argument("--oracle-dir", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--real-dir", type=Path, default=FIXTURES_DIR / "real_model_tracks")
    args = parser.parse_args()

    configs = (("Oracle labels", args.oracle_dir, style.ORACLE), ("Trained model", args.real_dir, style.REAL_MODEL))
    curves = {}
    for name, track_dir, _ in configs:
        pairs = answer_sets(args.questions_dir, track_dir)
        n_iou, iou = iou_acceptance(pairs, IOU_THRESHOLDS)
        n_dur, dur = tolerance_acceptance(pairs, TOLERANCES, "duration")
        curves[name] = {"iou": iou, "duration": dur}

    style.apply()
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True)
    for ax in (left, right):
        style.recede(ax)
    for name, _, colour in configs:
        common = dict(color=colour, linewidth=style.LINE_WIDTH, label=name, zorder=3, solid_capstyle="round")
        left.plot(IOU_THRESHOLDS, curves[name]["iou"], marker="o", markersize=style.MARKER_SIZE,
                  markeredgecolor=style.SURFACE, markeredgewidth=style.RING_WIDTH, **common)
        right.plot([t * 100 for t in TOLERANCES], curves[name]["duration"], drawstyle="steps-post", **common)

    left.axvline(metrics.HEADLINE_IOU_THRESHOLD, color=style.MUTED, linewidth=0.5, zorder=1)
    left.text(metrics.HEADLINE_IOU_THRESHOLD + 0.01, 1.04, "pre-registered 0.5", fontsize=7.5, color=style.MUTED)
    right.axvline(metrics.RELATIVE_DURATION_TOL * 100, color=style.MUTED, linewidth=0.5, zorder=1)
    right.text(metrics.RELATIVE_DURATION_TOL * 100 + 1.5, 1.04, "pre-registered 10%", fontsize=7.5, color=style.MUTED)

    left.set_xlabel("IoU threshold")
    left.set_xlim(0.08, 0.92)
    left.set_ylabel("Fraction of answers accepted")
    left.set_ylim(0, 1.12)
    left.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    left.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    left.set_title(f"Cited intervals (n={n_iou})", loc="left", fontsize=9, color=style.INK_SECONDARY)
    right.set_xlabel("Relative tolerance on duration (%)")
    right.set_xlim(0, 100)
    right.set_title(f"Durations (n={n_dur})", loc="left", fontsize=9, color=style.INK_SECONDARY)

    subjects = ", ".join(dev_subjects(args.questions_dir))
    fig.suptitle(f"Figure 3: Accuracy versus strictness - {subjects}", x=0.01, ha="left", fontsize=10)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2, handlelength=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT.with_suffix(".png"), dpi=style.DPI)
    plt.close(fig)

    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["panel", "x", *[name for name, _, _ in configs]])
        for i, t in enumerate(IOU_THRESHOLDS):
            writer.writerow(["iou_threshold", t, *[f"{curves[name]['iou'][i]:.4f}" for name, _, _ in configs]])
        for i, t in enumerate(TOLERANCES):
            writer.writerow(["duration_relative_tolerance", t, *[f"{curves[name]['duration'][i]:.4f}" for name, _, _ in configs]])

    at = {name: curves[name]["iou"][IOU_THRESHOLDS.index(metrics.HEADLINE_IOU_THRESHOLD)] for name, _, _ in configs}
    caption = [
        "**Figure 3: Accuracy versus strictness.** "
        f"Left: of the {n_iou} answers whose gold cites intervals, the fraction whose cited intervals reach each IoU "
        "threshold (matched mean IoU, PRD 7.3.3). Right: of the "
        f"{n_dur} duration answers, the fraction within each relative tolerance of the true total; the pre-registered "
        f"rule also has a {metrics.DURATION_ABS_TOL_S:g} s floor, left out of this sweep so the curve shows relative error alone. "
        "Oracle labels (blue) against the trained model (orange), both through the same reasoning layer, over "
        f"{subjects}. At the pre-registered IoU of {metrics.HEADLINE_IOU_THRESHOLD:g}: "
        + ", ".join(f"{name.lower()} {value:.0%}" for name, value in at.items())
        + ". A curve that stays high as the threshold tightens means misses are near misses; one that drops early means they are wild.",
        "",
        "Produced by `scripts/make_fig3.py`; numbers in `results/fig3_accuracy_vs_strictness.csv`.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(caption) + "\n", encoding="utf-8")

    for name, _, _ in configs:
        print(f"{name:14s} IoU@0.5 {at[name]:.2f}  duration@10% {curves[name]['duration'][10]:.2f}")
    print(f"-> {OUT.with_suffix('.png').relative_to(REPO_ROOT)} (+ .csv, .md)")


if __name__ == "__main__":
    main()
