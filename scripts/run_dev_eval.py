"""Measure the Phase 2 Member B exit criteria on the dev set (docs/TASKS.md).

Answers every dev subject's questions against that subject's oracle-style
fixture track, scores them, and writes a small summary to results/.

Usage:
    python scripts/run_dev_eval.py
    python scripts/run_dev_eval.py --label-noise 0.1 --out results/phase2_dev_eval_noise10.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ats.eval.dev import REPO_ROOT, run_dev_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "phase2_dev_eval.json"))
    args = parser.parse_args()

    report = run_dev_eval(label_noise=args.label_noise, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"subjects: {', '.join(report['subjects'])}   label noise: {args.label_noise}")
    for tier, row in sorted(report["by_tier"].items()):
        print(f"  tier {tier}: {row['accuracy']:.3f}  (n={row['n']})")
    print(f"  grounded accuracy @ IoU {report['iou_threshold']}: {report['grounded_accuracy']:.3f}  (n={report['n_grounded_rows']})")
    print(f"  overall macro accuracy: {report['overall_macro_accuracy']:.3f}")
    print(f"  withheld by validator: {report['n_rejected_by_validator']}   abstained: {report['n_abstained']}   empty: {report['n_empty_answers']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
