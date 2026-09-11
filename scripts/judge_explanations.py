"""Judge open-world explanations with Claude (docs/TASKS.md task 4B.4,
PRD §7.3.5), twice, and report the mean rubric score and the two runs'
agreement.

Claude Opus 5 is reached through OpenRouter's chat-completions endpoint
(docs/CITATIONS.md#claude-opus-5), with the rubric's JSON schema enforced as
structured output. Every judgment is cached in results/rubric_judgments.jsonl,
so a rerun only pays for what is missing and the reported numbers are
regenerable from the committed cache. The judged explanations are written to
results/rubric_items.jsonl.

The key is read from the OPENROUTER_API_KEY environment variable and never
written anywhere. Nothing is sent without --yes:

    python scripts/judge_explanations.py --dry-run
    python scripts/judge_explanations.py --yes
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

from probe_open_world import QUESTIONS as PROBE_QUESTIONS  # scripts/ is on sys.path when run directly

from ats.aggregate import load_track
from ats.eval.dev import FIXTURES_DIR, REPO_ROOT
from ats.eval.rubric import JUDGE_SCHEMA, RUBRIC, build_items, judge_prompt, parse_judgment, summarize_runs
from ats.serialize import read_question_set

# Haiku 4.5 rather than Opus 5 to keep the run cheap; see docs/TASKS.md 4B.4.
MODEL = "anthropic/claude-haiku-4.5"  # docs/CITATIONS.md#claude-haiku-45
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
KEY_VARIABLE = "OPENROUTER_API_KEY"
TIMEOUT_S = 300
RUNS = (1, 2)
RESULTS = REPO_ROOT / "results"
ITEMS_PATH = RESULTS / "rubric_items.jsonl"
JUDGMENTS_PATH = RESULTS / "rubric_judgments.jsonl"
SUMMARY_PATH = RESULTS / "rubric_summary.json"


class JudgeStop(Exception):
    """A failure that should stop the run: the next rerun resumes from the cache."""


def sources():
    probe = [{"question_id": f"probe{i}", "text": text} for i, text in enumerate(PROBE_QUESTIONS)]
    for subject in ("subj_synth_a", "subj_synth_b", "subj_synth_c"):
        questions = read_question_set(REPO_ROOT / "data" / "questions_dev" / f"{subject}.json")["questions"]
        yield f"{subject}/oracle", load_track(FIXTURES_DIR / f"track_{subject}.jsonl"), questions
    for subject in ("subj_real_a", "subj_real_b"):
        questions = read_question_set(REPO_ROOT / "data" / "questions_dev_v2" / f"{subject}.json")["questions"]
        for track, path in (
            ("oracle", FIXTURES_DIR / f"track_{subject}.jsonl"),
            ("real", FIXTURES_DIR / "real_model_tracks" / f"track_{subject}.jsonl"),
        ):
            yield f"{subject}/{track}", load_track(path), [*questions, *probe]


def load_cache() -> dict[tuple[int, str], dict]:
    if not JUDGMENTS_PATH.exists():
        return {}
    records = [json.loads(line) for line in JUDGMENTS_PATH.read_text(encoding="utf-8").splitlines() if line]
    # Only this judge's scores count: mixing two judges would make the
    # agreement figure meaningless.
    return {(r["run"], r["item_id"]): r for r in records if r["model"].startswith(MODEL)}


def judge(key: str, item: dict) -> dict:
    body = {
        "model": MODEL,
        "max_tokens": 16000,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": judge_prompt(item)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "rubric_judgment", "strict": True, "schema": JUDGE_SCHEMA},
        },
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        if error.code in (401, 403):
            raise SystemExit(f"OpenRouter rejected the key in {KEY_VARIABLE} (HTTP {error.code}): {detail}")
        if error.code == 402:
            raise SystemExit(f"OpenRouter reports insufficient credits (HTTP 402): {detail}")
        raise JudgeStop(f"HTTP {error.code}: {detail}")
    except (urllib.error.URLError, TimeoutError) as error:
        raise JudgeStop(f"connection failed: {error}")

    if "error" in data:
        raise JudgeStop(f"OpenRouter error: {data['error']}")
    choice = data["choices"][0]
    served_by = data.get("model", MODEL)
    content = choice["message"].get("content")
    if choice.get("finish_reason") in ("content_filter", "refusal") or not content:
        return {"error": f"no judgment (finish_reason={choice.get('finish_reason')})", "model": served_by}
    return parse_judgment(content) | {"model": served_by}


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge open-world explanations with Claude, twice.")
    parser.add_argument("--dry-run", action="store_true", help="Show the sample and the first prompt; send nothing.")
    parser.add_argument("--yes", action="store_true", help="Confirm sending the sample to OpenRouter.")
    args = parser.parse_args()

    items, abstained = build_items(list(sources()))
    cache = load_cache()
    todo = [(run, item) for run in RUNS for item in items if (run, item["item_id"]) not in cache]
    print(f"{len(items)} explanations to judge ({abstained} abstentions set aside), {len(RUNS)} runs each; "
          f"{len(todo)} calls still to make, {len(cache)} cached")
    RESULTS.mkdir(exist_ok=True)
    ITEMS_PATH.write_text("".join(json.dumps(i, sort_keys=True) + "\n" for i in items), encoding="utf-8")

    if args.dry_run:
        print("--- system prompt\n" + RUBRIC + "\n--- first item\n" + judge_prompt(items[0]))
        return
    if todo and not args.yes:
        raise SystemExit("add --yes to send these to OpenRouter")

    if todo:
        key = os.environ.get(KEY_VARIABLE)
        if not key:
            raise SystemExit(f"set {KEY_VARIABLE} first")
        with JUDGMENTS_PATH.open("a", encoding="utf-8") as out:
            for n, (run, item) in enumerate(todo, 1):
                try:
                    result = judge(key, item)
                except JudgeStop as error:
                    raise SystemExit(f"stopped at call {n}/{len(todo)}; rerun to resume: {error}")
                record = {"run": run, "item_id": item["item_id"], **result}
                out.write(json.dumps(record, sort_keys=True) + "\n")
                out.flush()
                print(f"{n}/{len(todo)} run {run} {item['item_id']}: "
                      + (result.get("error") or " ".join(str(result[c]) for c in ("cites_real_features", "features_support_conclusion", "conclusion_plausible"))))
        cache = load_cache()

    ids = {i["item_id"] for i in items}
    runs = {run: [r for (k, _), r in cache.items() if k == run and r["item_id"] in ids and "error" not in r] for run in RUNS}
    errors = [r for r in cache.values() if r["item_id"] in ids and "error" in r]
    summary = summarize_runs(runs[1], runs[2]) | {
        "judge_model": MODEL,
        "served_by": sorted({r["model"] for r in cache.values() if r["item_id"] in ids}),
        "n_sampled": len(items),
        "n_abstentions_set_aside": abstained,
        "n_judgments_without_score": len(errors),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"-> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
