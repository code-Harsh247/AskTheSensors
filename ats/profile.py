"""PRD Sec 6.1 minimum resource-cost reporting (docs/TASKS.md task 4A.2):
parameter count, on-disk size, peak resident memory, and per-query latency
(p50/p95) for the recognition backbone, on the frozen target device
(docs/TASKS.md Sec 0). Emits a schema-valid CostReport
(schemas/cost_report.schema.json).

`time_calls` and `peak_rss_mb` are deliberately generic, not tied to
ActivityCNN -- docs/TASKS.md task 4A.2 notes "A owns the harness; B
instruments the interface layer with it," since PRD Sec 6.1's per-query
latency spans the whole answer pipeline (recognition + reasoning + SLM),
and only Member B's code can time that end-to-end (task 4B.6).

A "query" here is one recognition call: classifying a single already-
windowed 4-second signal segment, matching the per-window inference
`ats.recognize.build_track` performs in production. This is the piece
Member A's own code owns; it is not the full PRD Sec 6.1 per-query number,
which also includes B's reasoning/SLM layer.

Usage: python -m ats.profile --config full [--model PATH] [--n-queries N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import psutil
import torch

from ats.contracts import validate_cost_report
from ats.model import ActivityCNN, N_CHANNELS, count_parameters, predict_probs
from ats.resample import TARGET_HZ
from ats.windowing import WINDOW_LENGTH_S

REPO_ROOT = Path(__file__).resolve().parent.parent

# Frozen docs/TASKS.md Sec 0 -- stated verbatim, not re-derived here.
# Corrected 2026-09-11: the original entry (5800H, 16GB) named a different
# machine than the one actually producing these numbers.
TARGET_DEVICE = (
    "Laptop CPU: AMD Ryzen 7 4800H (8C/16T), 8GB RAM, Windows 11 64-bit, "
    "single-process CPU inference (no GPU)"
)

# One window's timestep count at the frozen window length / sample rate
# (docs/TASKS.md Sec 0): matches the shape ats.windowing.window_to_tensor
# actually produces, rather than a hardcoded guess.
N_TIMESTEPS = int(round(WINDOW_LENGTH_S * TARGET_HZ)) + 1
DEFAULT_N_QUERIES = 200


def time_calls(fn: Callable[[], None], n: int) -> tuple[float, float]:
    """Call `fn` n times and return (p50_ms, p95_ms) wall-clock latency per
    call, nearest-rank percentile over the sorted samples. Generic on
    purpose -- reusable for 4B.6, not specific to the CNN."""
    samples_ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms.sort()

    def pct(p: float) -> float:
        idx = int(round(p * (len(samples_ms) - 1)))
        return samples_ms[idx]

    return pct(0.50), pct(0.95)


def peak_rss_mb() -> float:
    """Peak resident memory used by this process so far, in MB. Windows'
    psutil backend reports a true peak (`peak_wset`); other platforms don't
    track one separately, so this falls back to current RSS there --- an
    underestimate, but the frozen target device (docs/TASKS.md Sec 0) is
    Windows, where the true peak is used."""
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is not None:
        return peak / (1024 * 1024)
    return info.rss / (1024 * 1024)


def profile_recognition(model_path: Path, n_queries: int) -> dict:
    """Build a CostReport (minus config_id/target_device, filled in by the
    caller) for one trained ActivityCNN checkpoint."""
    model = ActivityCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((1, N_CHANNELS, N_TIMESTEPS)).astype(np.float32))

    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime; first call always reads 0.0
    p50_ms, p95_ms = time_calls(lambda: predict_probs(model, x), n_queries)
    cpu_pct = proc.cpu_percent(interval=None)

    return {
        "params": count_parameters(model),
        "disk_mb": model_path.stat().st_size / (1024 * 1024),
        "peak_rss_mb": peak_rss_mb(),
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "cpu_pct": cpu_pct,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ats.profile",
        description="Emit a schema-valid cost report for one model configuration (docs/TASKS.md task 4A.2).",
    )
    parser.add_argument("--config", default="full", help="Config name, matching models/<config>/ (docs/TASKS.md task 5A.1).")
    parser.add_argument("--model", default=None, help="Path to weights; defaults to models/<config>/activity_cnn.pt.")
    parser.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES, help="Number of single-window inference calls to time.")
    parser.add_argument("--out", default=None, help="Output path; defaults to results/cost_report_<config>.json.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    model_path = Path(args.model) if args.model else REPO_ROOT / "models" / args.config / "activity_cnn.pt"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"{model_path} does not exist -- place trained weights there first (see scripts/train_cnn.py)"
        )

    report = profile_recognition(model_path, args.n_queries)
    report["config_id"] = args.config
    report["target_device"] = TARGET_DEVICE
    validate_cost_report(report)

    out_path = Path(args.out) if args.out else REPO_ROOT / "results" / f"cost_report_{args.config}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
