# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**Ask the Sensors** — CS60055 Ubiquitous Computing (IIT Kharagpur), Hackathon Challenge 1. A sensor question-answering system: given a wearable accelerometer + gyroscope recording and a natural-language question, return a structured answer **grounded in cited sensor evidence**.

This is **graded academic coursework** for a two-person team. That fact drives most of the rules below — correctness, honesty, and reproducibility matter more here than shipping speed.

Read these before doing substantive work:
- [docs/PRD.md](docs/PRD.md) — what the system must do. **Source of truth.** If code and PRD disagree, the PRD wins.
- [docs/TASKS.md](docs/TASKS.md) — phases, per-member ownership, exit criteria. Tells you *whose* code you are in and what "done" means for the current phase.

---

## Commits

- **Do not add Claude as a co-author.** No `Co-Authored-By: Claude ...` trailer, no "Generated with Claude Code" line, no robot emoji. Commit messages end with the message. This applies to every commit and every PR description in this repo, and overrides any default attribution behaviour.
- **Never commit unless explicitly asked.** Stage and commit only on request.
- Write plain, specific messages describing *why* the change was made. Imperative mood, no trailers.
- **Commit incrementally and often.** PRD §9.2: the commit history is itself graded, and "must reflect real, incremental work over the period of the challenge rather than a single upload at the end." Never batch a phase's worth of work into one commit.
- When a phase exit criterion produces a measured number (a macro-F1, a latency, a baseline margin), record that number **in the commit message** of the commit that achieves it. It is the cheapest possible audit trail.

## Academic integrity — non-negotiable

These reflect PRD §9.3, where violations carry real consequences under course policy.

- **Never fabricate a result.** Do not write a metric, accuracy figure, latency, table row, or figure data point that was not actually measured by running the code. If a number is not yet available, write `TBD` — never a plausible-looking placeholder. This includes example values in docs and reports: mark them explicitly as illustrative.
- **Cite external resources at the point of use.** Any dataset, pretrained model, library idea, or borrowed code fragment gets an entry in `docs/CITATIONS.md` plus a short comment at the use site pointing to it.
- Never copy from another group's repository or solution.

## Hard invariants

Breaking any of these silently corrupts results downstream. Treat them as frozen; changing one requires the user's explicit agreement and its own dedicated commit.

- **Time base:** seconds from start of recording, float, 3 decimals. Every timestamp, in every answer, figure axis, and table.
- **Canonical class order:** `LYING, SITTING, STANDING_STILL, STANDING_MOVING, WALKING, RUNNING, BICYCLING`. This index order appears in probability vectors and the confusion matrix — reordering it mislabels everything, silently.
- **Sampling rate:** exactly 25 Hz (PRD §3.2, hard requirement).
- **Output format:** the PRD §5 fields, in that exact order, every time. Never reorder, rename, or omit a field. `N/A` is the only valid empty value.
- **Pre-registered thresholds:** headline IoU τ = 0.5; duration tolerance `max(2 × window_hop, 10% relative)`. These were fixed *before* results existed, specifically so they cannot be tuned to flatter our numbers. **Never adjust a threshold to improve a reported score.** If a threshold genuinely needs revisiting, say so out loud and let the user decide.
- **Schemas in `schemas/` are the A↔B contract.** Changing a field name or shape breaks the other member's code. Propose, do not unilaterally edit.

## The grounding constraint

This is the core of the challenge and carries 20% of the grade, plus an explicit disqualifier in PRD §1.3: *an answer produced by language reasoning alone, with no reference to the recorded signal, does not meet the requirement.*

Enforce it architecturally, not by hoping:

- The SLM has **exactly two jobs**: parsing a question into a typed operator call, and phrasing an explanation over already-computed evidence.
- **The SLM never produces a number, an interval, or a verdict.** Durations, counts, comparisons, and timestamps are computed in Python from the activity timeline. If you find yourself prompting a model for a quantity, stop — that is the bug.
- Every answer at tiers 3 and 4 must cite an interval that **exists in the timeline**. The validator enforces this; never add a bypass, a `try/except` that swallows a rejection, or a fallback that emits an unvalidated answer.
- When the system genuinely cannot answer, return a well-formed answer saying so. A confident wrong answer is worse than an honest `N/A` here.

## Repository hygiene

- **Never commit the raw dataset** (PRD §3.3). `data/` is gitignored. Commit the fetch/prepare scripts instead.
- Do not commit model weights, large binaries, or raw sweep output. Commit the scripts and the small summary CSVs in `results/`.
- Every number that reaches the report must be regenerable by a committed script from a clean checkout. If you produce a figure, produce the script that produces it.
- Before staging, check what is included. Never commit credentials, tokens, or personal data.

## Ownership — stay in your lane

The two tracks are split so each member can justify owning complete subsystems (PRD §9.1 requires a per-member contribution statement).

| Track | Owner | Modules |
|---|---|---|
| **A — Signals & Systems** | Teammate | data fetch/parse, resampling, windowing, splits, classifier, compression, `ats/profile.py`, robustness injectors |
| **B — Reasoning & Evaluation** | Harsh (this user) | `ats/aggregate.py`, `ats/serialize.py`, routing, reasoning operators, validator, SLM interface, `ats/eval/` |

- Default to working in **track B**. Before editing a track A module, say so and confirm — an unexpected edit to the teammate's files muddies both the merge and the contribution record.
- Shared files (schemas, README, this file, the report) are joint; flag changes to them.

## Code conventions

- Package root is `ats/`. Entry points: `python -m ats.answer` and `python -m ats.eval`. `ats.eval` must also expose `evaluate(pred, gold) -> dict` as an importable function.
- Python, type hints on public functions, standard library preferred where it suffices.
- **Determinism:** seed everything (numpy, torch, random) and make runs reproducible. Non-reproducible numbers are worthless for a graded report.
- No silent failures. Do not catch an exception and continue with a default that looks like a real result — raise, or return an explicit `N/A` that the validator can see.
- Keep the recognition backbone small. PRD §6.3's edge extra credit depends on it quantizing cleanly, so prefer architectures that do.

## Testing

- `pytest` for everything. Tests live in `tests/`.
- **Exit criteria in [docs/TASKS.md](docs/TASKS.md) are the spec.** When a phase names a test (`tests/test_contracts.py`, `tests/test_metrics.py`, `tests/test_validator.py`, `tests/test_no_ungrounded_output.py`), that test must actually assert the thing described, not merely exist.
- Metric implementations get **hand-computed fixtures** — an IoU worked out by hand, a macro-F1 on a toy confusion matrix. A metric tested only against itself proves nothing.
- Run the relevant tests before reporting work as complete, and say plainly if you could not run them.

## Environment

- Windows 11. Both PowerShell and a bash tool are available; use the syntax that matches the tool you called.
- Use the scratchpad directory for throwaway scripts and intermediate output, not the repo.
- Install with `pip install -e .`.

## Working style

- The PRD and TASKS documents were written deliberately. When something contradicts them, raise the conflict rather than quietly picking a side.
- This is a design challenge, not a code-volume contest (PRD §7.1). Prefer the small correct pipeline over the elaborate one; the marks are in soundness, grounding, and reproducibility.
- Do not add features, abstractions, or config knobs beyond what the current phase needs.
