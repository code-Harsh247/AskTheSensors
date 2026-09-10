"""CLI entry point: python -m ats.oracle --subject <id> --out track.jsonl

Emits a schema-valid `window_track` for one ExtraSensory subject, built from
ground-truth main-activity labels rather than a trained model -- the Phase 1
"unblocking deliverable" (docs/TASKS.md task 1A.8) that lets Member B build
the entire reasoning/evaluation stack without waiting on the real classifier.
Feature values are still computed from the subject's real, resampled signal
(see ats/windowing.py), so only the *label* is ground truth -- exactly what a
real classifier would eventually replace.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ats.contracts import CANONICAL_CLASSES, validate_window_track
from ats.ingest import load_subject
from ats.resample import globalize_subject
from ats.windowing import HOP_S, WINDOW_LENGTH_S, feature_summary, make_windows

DEFAULT_DATA_DIR = "data/raw"


def _label_for_window(t_start: float, t_end: float, spans) -> str | None:
    """A window's ground truth is the activity of the burst it falls inside.
    Windows are shorter than a burst by construction (WINDOW_LENGTH_S well
    under one ~20s burst), so a window that straddles two spans, or falls
    entirely in the dead time between bursts, has no single ground truth and
    is skipped rather than guessed."""
    for activity, span_start, span_end in spans:
        if span_start <= t_start and t_end <= span_end:
            return activity
    return None


def _probs_for(label: str, soften_confidence: bool, rng: random.Random) -> list[float]:
    true_index = CANONICAL_CLASSES.index(label)
    if soften_confidence:
        confidence = 0.70 + rng.uniform(0.0, 0.25)
    else:
        confidence = 1.0
    remainder = (1.0 - confidence) / (len(CANONICAL_CLASSES) - 1)
    probs = [remainder] * len(CANONICAL_CLASSES)
    probs[true_index] = confidence
    return probs


def build_track(
    subject_id: str,
    data_dir: str | Path,
    *,
    label_noise: float = 0.0,
    soften_confidence: bool = False,
    drop_windows: float = 0.0,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    subject = load_subject(data_dir, subject_id)
    globalized = globalize_subject(subject)
    windows = make_windows(globalized, window_s=WINDOW_LENGTH_S, hop_s=HOP_S)

    entries: list[dict] = []
    for i, w in enumerate(windows):
        true_label = _label_for_window(w.t_start, w.t_end, globalized.label_spans)
        if true_label is None:
            continue
        if drop_windows and rng.random() < drop_windows:
            continue

        label = true_label
        if label_noise and rng.random() < label_noise:
            label = rng.choice([c for c in CANONICAL_CLASSES if c != true_label])

        entry = {
            "window_id": f"w{i:06d}",
            "t_start": round(w.t_start, 3),
            "t_end": round(w.t_end, 3),
            "probs": _probs_for(label, soften_confidence, rng),
            "coverage": round(w.coverage, 4),
            "feature_summary": feature_summary(w),
            "model_id": "oracle",
        }
        validate_window_track(entry)
        entries.append(entry)
    return entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ats.oracle",
        description="Emit a ground-truth window_track for one ExtraSensory subject "
        "(the Phase 1 unblocking deliverable -- see docs/TASKS.md task 1A.8).",
    )
    parser.add_argument("--subject", required=True, help="Subject UUID.")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory passed to scripts/fetch_data.py --out (archives live under its _meta/ subdir).",
    )
    parser.add_argument("--out", required=True, help="Path to write the window_track JSONL to.")
    parser.add_argument("--label-noise", type=float, default=0.0, help="Fraction of windows given a wrong label.")
    parser.add_argument(
        "--soften-confidence", action="store_true", help="Emit realistic probability spreads instead of one-hot."
    )
    parser.add_argument("--drop-windows", type=float, default=0.0, help="Fraction of windows omitted entirely.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the noise/drop/confidence randomness.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    entries = build_track(
        args.subject,
        args.data_dir,
        label_noise=args.label_noise,
        soften_confidence=args.soften_confidence,
        drop_windows=args.drop_windows,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"wrote {len(entries)} windows -> {out}")


if __name__ == "__main__":
    main()
