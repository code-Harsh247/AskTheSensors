"""Phase 4 router ablation (docs/TASKS.md Phase 4 exit criteria): the SLM
router against the rule router.

Three measurements per router:
  - end-to-end QA accuracy on the frozen dev set (routing plus everything
    downstream);
  - routing accuracy on held-out phrasings (tests/fixtures/routing_heldout.json),
    reported separately for questions from the brief and questions written by
    Member B;
  - wall-clock routing latency per question on the target laptop. Informal:
    the official cost numbers come from ats/profile.py (task 4B.6).

Usage:  python scripts/run_router_ablation.py [--out results/phase4_router_ablation.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from ats.eval.delta import COMPATIBLE_OPERATORS
from ats.eval.dev import QUESTIONS_DIR, REPO_ROOT, dev_subjects, run_dev_eval
from ats.routing import OperatorCall, route
from ats.serialize import read_question_set
from ats.slm import MODEL_ID, MODEL_REVISION, SLMRouter

HELDOUT = REPO_ROOT / "tests" / "fixtures" / "routing_heldout.json"


def call_matches(call: OperatorCall, expected: dict[str, Any]) -> bool:
    if call.op != expected["op"]:
        return False
    if call.op == "open_world" and call.predicate != expected.get("predicate"):
        return False
    if "activities" in expected and set(call.activities) != set(expected["activities"]):
        return False
    if "time_s" in expected and call.time_s != expected["time_s"]:
        return False
    return True


def timed(router: Callable[[str], OperatorCall], text: str) -> tuple[OperatorCall, float]:
    start = time.perf_counter()
    call = router(text)
    return call, (time.perf_counter() - start) * 1000.0


def measure(name: str, router: Callable[[str], OperatorCall]) -> dict[str, Any]:
    dev_questions = [
        q for s in dev_subjects() for q in read_question_set(QUESTIONS_DIR / f"{s}.json")["questions"]
    ]
    latencies: list[float] = []

    dev_routed = 0
    for question in dev_questions:
        call, ms = timed(router, question["text"])
        latencies.append(ms)
        dev_routed += call.op in COMPATIBLE_OPERATORS[question["gold"]["question_type"]]

    heldout = json.loads(HELDOUT.read_text(encoding="utf-8"))["items"]
    by_source: dict[str, list[bool]] = {}
    misroutes: list[dict[str, Any]] = []
    for item in heldout:
        call, ms = timed(router, item["text"])
        latencies.append(ms)
        ok = call_matches(call, item["expected"])
        by_source.setdefault(item["source"], []).append(ok)
        if not ok:
            misroutes.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "expected": item["expected"],
                    "got": {"op": call.op, "activities": list(call.activities), "time_s": call.time_s, "predicate": call.predicate},
                }
            )

    qa = run_dev_eval(router=router)
    return {
        "router": name,
        "dev_routing_accuracy": dev_routed / len(dev_questions),
        "dev_qa_macro_accuracy": qa["overall_macro_accuracy"],
        "dev_qa_by_tier": {tier: row["accuracy"] for tier, row in qa["by_tier"].items()},
        "dev_grounded_accuracy": qa["grounded_accuracy"],
        "heldout_routing_accuracy": {
            source: {"n": len(oks), "accuracy": sum(oks) / len(oks)} for source, oks in by_source.items()
        },
        "heldout_misroutes": misroutes,
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": statistics.quantiles(latencies, n=20)[18],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the SLM router with the rule router.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "phase4_router_ablation.json")
    args = parser.parse_args()

    slm = SLMRouter()
    start = time.perf_counter()
    slm(("warm-up: what activity is the user performing?"))
    load_s = time.perf_counter() - start
    slm.n_parsed = slm.n_fallback = 0
    slm.failures.clear()

    rules = measure("rules", route)
    model = measure("slm", slm)
    model["model"] = {"id": MODEL_ID, "revision": MODEL_REVISION, "load_and_first_call_s": load_s}
    model["parsed_by_slm"] = slm.n_parsed
    model["rule_fallbacks"] = slm.n_fallback
    model["fallback_examples"] = slm.failures[:10]

    report = {
        "note": "Latency is informal wall-clock on the target laptop (docs/TASKS.md section 0); official cost figures come from ats/profile.py.",
        "rules": rules,
        "slm": model,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for row in (rules, model):
        heldout = ", ".join(f"{s} {v['accuracy']:.2f} (n={v['n']})" for s, v in row["heldout_routing_accuracy"].items())
        print(
            f"{row['router']:5s}  dev routing {row['dev_routing_accuracy']:.3f}  dev QA macro {row['dev_qa_macro_accuracy']:.3f}  "
            f"held-out {heldout}  latency p50 {row['latency_ms_p50']:.1f} ms, p95 {row['latency_ms_p95']:.1f} ms"
        )
    print(f"SLM: parsed {model['parsed_by_slm']}, fell back to rules {model['rule_fallbacks']}; load + first call {load_s:.1f} s")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
