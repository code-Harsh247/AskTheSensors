"""Time the question-answering layer with Member A's profiler (docs/TASKS.md
task 4B.6). PRD §6.1's per-query latency spans recognition and answering;
`python -m ats.profile` times recognition per window, and this times the
rest: building the timeline once per recording, then routing, answering and
validating each question, optionally with the SLM parser.

The frozen target device is Member A's laptop (docs/TASKS.md §0), so only a
run there gives the reported numbers. The machine a run actually used is
recorded next to the target, so a run elsewhere is never mistaken for one.

Usage:  python scripts/profile_pipeline.py [--router slm] [--out PATH]
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
from pathlib import Path

import psutil

from ats.aggregate import build_timeline, load_track
from ats.answer import answer_question
from ats.eval.dev import FIXTURES_DIR, REPO_ROOT
from ats.profile import TARGET_DEVICE, peak_rss_mb, time_calls
from ats.routing import route
from ats.serialize import read_question_set

# The larger real subject: about 41,000 windows over 46 hours.
SUBJECT = "subj_real_b"
TIMELINE_RUNS = 5
ANSWER_PASSES = 3  # each question answered this many times
SLM_QUESTIONS = 10  # about 5 s each on the development laptop


def machine() -> str:
    return (
        f"{platform.processor() or platform.machine()}, {psutil.cpu_count(logical=False)}C/"
        f"{psutil.cpu_count()}T, {psutil.virtual_memory().total / 2**30:.0f} GB RAM, {platform.system()} {platform.release()}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--router", choices=["rules", "slm"], default="rules")
    parser.add_argument("--out", default=None, help="Defaults to results/pipeline_latency_<router>.json.")
    args = parser.parse_args(argv)

    windows = load_track(FIXTURES_DIR / "real_model_tracks" / f"track_{SUBJECT}.jsonl")
    questions = read_question_set(REPO_ROOT / "data" / "questions_dev_v2" / f"{SUBJECT}.json")["questions"]
    timeline_p50, timeline_p95 = time_calls(lambda: build_timeline(windows), TIMELINE_RUNS)
    timeline = build_timeline(windows)

    router = route
    n = ANSWER_PASSES * len(questions)
    if args.router == "slm":
        from ats.slm import SLMRouter

        router = SLMRouter()
        router(questions[0]["text"])  # load the model before timing
        n = SLM_QUESTIONS
    cycle = itertools.cycle(questions)
    answer_p50, answer_p95 = time_calls(lambda: answer_question(next(cycle), timeline, windows, router), n)

    report = {
        "target_device": TARGET_DEVICE,
        "measured_on": machine(),
        "router": args.router,
        "subject": SUBJECT,
        "n_windows": len(windows),
        "timeline_build_ms": {"p50": timeline_p50, "p95": timeline_p95, "runs": TIMELINE_RUNS},
        "per_question_ms": {"p50": answer_p50, "p95": answer_p95, "calls": n},
        "peak_rss_mb": peak_rss_mb(),
    }
    out = Path(args.out) if args.out else REPO_ROOT / "results" / f"pipeline_latency_{args.router}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
