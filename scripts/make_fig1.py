"""Figure 1 (PRD 7.4.1; docs/TASKS.md 5B.3): accuracy by question type, the
trained model against oracle labels, with a macro-averaged overall bar.

Answers every question set in --questions-dir against the oracle tracks and
the real-model tracks and scores them with ats.eval.evaluate. Writes:
    results/fig1_accuracy_by_question_type.png
    results/fig1_accuracy_by_question_type.csv   (the table view)
    results/fig1_accuracy_by_question_type.md    (caption with each group's correctness rule)

Usage: python scripts/make_fig1.py [--questions-dir data/questions_dev_v2]
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

from ats.eval import QUESTION_TYPES, evaluate  # noqa: E402
from ats.eval import metrics  # noqa: E402
from ats.eval.curves import answer_sets  # noqa: E402
from ats.eval.dev import FIXTURES_DIR, dev_subjects  # noqa: E402

OUT = REPO_ROOT / "results" / "fig1_accuracy_by_question_type"

RULES = {
    "identification": "exact match of the named activity",
    "verification": "exact match of the yes/no verdict",
    "duration": f"within max({metrics.DURATION_ABS_TOL_S:g} s, {metrics.RELATIVE_DURATION_TOL:.0%}) of the true total (pre-registered)",
    "count": f"within {metrics.COUNT_ABS_TOL:g} bout of the true count",
    "comparison": "exact match of the activity that took longer, or 'Equal'",
    "grounding": f"cited intervals reach matched IoU >= {metrics.HEADLINE_IOU_THRESHOLD:g} with the true intervals (pre-registered)",
    "open_world": "exact match of the verdict to the reference behaviour",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Figure 1.")
    parser.add_argument("--questions-dir", type=Path, default=REPO_ROOT / "data" / "questions_dev_v2")
    parser.add_argument("--oracle-dir", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--real-dir", type=Path, default=FIXTURES_DIR / "real_model_tracks")
    args = parser.parse_args()

    configs = (("Oracle labels", args.oracle_dir, style.ORACLE), ("Trained model", args.real_dir, style.REAL_MODEL))
    reports = {}
    for name, track_dir, _ in configs:
        pairs = answer_sets(args.questions_dir, track_dir)
        reports[name] = evaluate({"answers": [a for _, a in pairs]}, {"questions": [q for q, _ in pairs]})

    first = reports[configs[0][0]]
    types = [t for t in QUESTION_TYPES if t in first["by_question_type"]]
    n_questions = first["n_graded"]

    def accuracy(report: dict, group: str) -> float:
        return report["overall_macro_accuracy"] if group == "overall" else report["by_question_type"][group]["accuracy"]

    groups = types + ["overall"]
    style.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    style.recede(ax)
    xs = [i if g != "overall" else i + 0.5 for i, g in enumerate(groups)]
    width, gap = 0.18, 0.02
    for k, (name, _, colour) in enumerate(configs):
        offset = (k - 0.5) * (width + gap)
        values = [accuracy(reports[name], g) for g in groups]
        ax.bar([x + offset for x in xs], values, width=width, color=colour, label=name, zorder=2)
        if name == "Trained model":
            for x, v in zip(xs, values):
                ax.text(x + offset, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5, color=style.INK_SECONDARY)

    labels = [f"{g.replace('_', '-')}\nn={first['by_question_type'][g]['n']}" for g in types] + ["overall\n(macro)"]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Accuracy")
    ax.axvline(len(types) - 0.25, color=style.GRID, linewidth=0.5)
    subjects = ", ".join(dev_subjects(args.questions_dir))
    ax.set_title(
        f"Figure 1: Accuracy by question type\n{n_questions} questions over {subjects}",
        loc="left",
        fontsize=10,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, handlelength=1.0)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=style.DPI)
    plt.close(fig)

    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_type", "n", "correctness_rule", *[name for name, _, _ in configs]])
        for g in groups:
            n = first["by_question_type"][g]["n"] if g != "overall" else n_questions
            rule = RULES.get(g, "macro-average of the per-type accuracies (PRD 7.3)")
            writer.writerow([g, n, rule, *[f"{accuracy(reports[name], g):.4f}" for name, _, _ in configs]])

    caption = [
        "**Figure 1: Accuracy by question type.** "
        f"{n_questions} questions over {subjects} (`{args.questions_dir.relative_to(REPO_ROOT).as_posix()}`), answered "
        "from oracle labels (blue) and from the trained model's predictions (orange); the same reasoning layer answers both. "
        "Overall is the macro-average of the per-type accuracies, so plentiful easy types cannot dominate it (PRD 7.3). "
        "Correctness rule by group:",
        "",
        *[f"- **{t.replace('_', '-')}**: {RULES[t]}." for t in types],
        "",
        "Produced by `scripts/make_fig1.py`; numbers in `results/fig1_accuracy_by_question_type.csv`.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(caption) + "\n", encoding="utf-8")

    for name, _, _ in configs:
        print(f"{name:14s} " + "  ".join(f"{g}={accuracy(reports[name], g):.2f}" for g in groups))
    print(f"-> {OUT.with_suffix('.png').relative_to(REPO_ROOT)} (+ .csv, .md)")


if __name__ == "__main__":
    main()
