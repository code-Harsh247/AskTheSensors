"""Figure 4 (PRD Sec 6.3, docs/TASKS.md task 5A.4): accuracy vs. cost across
every compressed configuration, with the non-dominated (Pareto) frontier
drawn via Member B's `ats.eval.pareto.pareto_frontier`.

Reads results/pareto.csv (scripts/sweep.py). Writes:
    results/fig4_pareto.png
    results/fig4_pareto.csv   (the table view, same rows as pareto.csv)
    results/fig4_pareto.md    (caption: cost axis, accuracy definition, frontier rule)

Usage: python scripts/make_fig4.py [--pareto results/pareto.csv] [--cost-axis disk_mb]
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
from ats.eval.pareto import pareto_frontier  # noqa: E402

DEFAULT_PARETO = REPO_ROOT / "results" / "pareto.csv"
OUT = REPO_ROOT / "results" / "fig4_pareto"

COST_LABELS = {
    "disk_mb": "On-disk size (MB)",
    "latency_p50_ms": "Median per-query latency (ms)",
    "peak_rss_mb": "Peak resident memory (MB)",
    "params": "Parameter count",
}
COST_LABELS_LOWER = {
    "disk_mb": "on-disk size",
    "latency_p50_ms": "median per-query latency",
    "peak_rss_mb": "peak resident memory",
    "params": "parameter count",
}


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key != "config_id" and key != "target_device":
                row[key] = float(value)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto", default=str(DEFAULT_PARETO))
    parser.add_argument("--cost-axis", default="disk_mb", choices=list(COST_LABELS))
    args = parser.parse_args(argv)

    rows = _read_rows(Path(args.pareto))
    if len(rows) < 2:
        raise ValueError(f"{args.pareto} has fewer than 2 rows -- run scripts/sweep.py first")

    frontier = pareto_frontier(rows, cost=args.cost_axis, value="accuracy")
    frontier_ids = {r["config_id"] for r in frontier}

    style.apply()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    style.recede(ax)

    xs = [r[args.cost_axis] for r in rows]
    ys = [r["accuracy"] for r in rows]
    colors = [style.ORACLE if r["config_id"] in frontier_ids else style.MUTED for r in rows]
    ax.scatter(xs, ys, c=colors, s=40, zorder=3)
    for r in rows:
        ax.annotate(
            r["config_id"],
            (r[args.cost_axis], r["accuracy"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=style.INK_SECONDARY,
        )

    frontier_sorted = sorted(frontier, key=lambda r: r[args.cost_axis])
    ax.plot(
        [r[args.cost_axis] for r in frontier_sorted],
        [r["accuracy"] for r in frontier_sorted],
        color=style.ORACLE,
        linewidth=style.LINE_WIDTH,
        zorder=2,
        label="Non-dominated (Pareto frontier)",
    )

    ax.set_xlabel(COST_LABELS[args.cost_axis])
    ax.set_ylabel("Accuracy (overall macro, ats.eval.evaluate)")
    ax.set_title("Figure 4: Accuracy vs. cost across compressed configurations", loc="left", fontsize=10)
    ax.legend(loc="lower right", handlelength=1.2)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=style.DPI)
    plt.close(fig)

    shutil.copy(args.pareto, OUT.with_suffix(".csv"))

    full_row = next((r for r in rows if r["config_id"] == "full"), None)
    tradeoff_note = ""
    if full_row is not None:
        cheapest = min(rows, key=lambda r: r[args.cost_axis])
        cost_cut = 1 - cheapest[args.cost_axis] / full_row[args.cost_axis]
        acc_delta = cheapest["accuracy"] - full_row["accuracy"]
        direction = "a gain of" if acc_delta > 0 else ("a loss of" if acc_delta < 0 else "no change in")
        tradeoff_note = (
            f"The cheapest config (`{cheapest['config_id']}`) cuts {COST_LABELS_LOWER[args.cost_axis]} "
            f"by {cost_cut:.0%} relative to `full`, for {direction} {abs(acc_delta):.4f} accuracy. "
        )
        dominates_full = "full" not in frontier_ids and any(r["config_id"] != "full" for r in frontier)
        if dominates_full:
            beaters = sorted(
                r["config_id"] for r in frontier if r[args.cost_axis] <= full_row[args.cost_axis] and r["accuracy"] >= full_row["accuracy"]
            )
            verb = "matches or beats" if len(beaters) == 1 else "match or beat"
            tradeoff_note += (
                f"`full` itself is dominated here: {', '.join(f'`{b}`' for b in beaters)} {verb} it on both "
                f"{COST_LABELS_LOWER[args.cost_axis]} and accuracy at once, on this 34-question dev set. "
            )

    caption = [
        "**Figure 4: Accuracy vs. cost across compressed configurations.** "
        f"Cost axis: {COST_LABELS_LOWER[args.cost_axis]} (`results/pareto.csv` also carries "
        "params, latency p50/p95, and peak RSS as additional cost columns per docs/TASKS.md task 5A.1). "
        "Accuracy is `ats.eval.evaluate`'s overall macro accuracy over the frozen "
        f"`data/questions_dev_v2` question set -- the same pre-registered rules as Figure 1 "
        f"(IoU >= {metrics.HEADLINE_IOU_THRESHOLD:g} for grounding, "
        f"max({metrics.DURATION_ABS_TOL_S:g}s, {metrics.RELATIVE_DURATION_TOL:.0%}) for duration, "
        f"+/-{metrics.COUNT_ABS_TOL:g} bout for count). "
        "Blue points and the connecting line are the non-dominated (Pareto) frontier "
        "(`ats.eval.pareto.pareto_frontier`): no other configuration matches or beats them on both "
        "cost and accuracy simultaneously. "
        f"{tradeoff_note}"
        "Compressed configs are quantized (int8, post-training static quantization) and "
        "structurally pruned (30%/60% of Conv1d channels removed by L1-norm rank, no fine-tuning); "
        "see `ats/compress.py`.",
        "",
        "Produced by `scripts/make_fig4.py`; numbers in `results/fig4_pareto.csv`.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(caption) + "\n", encoding="utf-8")

    print(f"non-dominated: {sorted(frontier_ids)}")
    print(f"-> {OUT.with_suffix('.png').relative_to(REPO_ROOT)} (+ .csv, .md)")


if __name__ == "__main__":
    main()
