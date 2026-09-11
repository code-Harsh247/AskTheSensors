"""CLI entry point: python -m ats.recognize --subject <id> --model-path models/full/activity_cnn.pt --out track.jsonl

Emits a schema-valid `window_track` for one ExtraSensory subject using the
trained `ats.model.ActivityCNN` (docs/TASKS.md task 2A.4) -- the real
counterpart to `ats/oracle.py`'s ground-truth track. This is what
`ats/answer.py --model` needs for Phase 3's swap (replacing the oracle track
with the real classifier's output), and what feeds
`scripts/oracle_delta.py`'s comparison.

Unlike `ats/oracle.py`, every window `ats.windowing.make_windows` produces
gets a prediction here, regardless of its `coverage` value -- a live system
cannot decline to answer just because a window is imperfectly covered, and
`ats/aggregate.py` already has its own coverage-based trust threshold for
deciding what to use downstream. Skipping low-coverage windows here would
just duplicate that policy in the wrong layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ats.contracts import validate_window_track
from ats.ingest import load_subject
from ats.model import ActivityCNN, predict_probs
from ats.resample import globalize_subject
from ats.windowing import HOP_S, WINDOW_LENGTH_S, feature_summary, make_windows, window_to_tensor

DEFAULT_DATA_DIR = "data/raw"
DEFAULT_MODEL_PATH = "models/full/activity_cnn.pt"


def load_model(model_path: str | Path) -> torch.nn.Module:
    """Loads `models/full/activity_cnn.pt`'s plain state-dict format
    unchanged (the graders' entry point, `ats.answer`, must keep working
    exactly as before). Compressed configs from `ats/compress.py` (task
    5A.1) save the whole module instead -- pruning changes the architecture
    (narrower channels), and quant8 is saved via TorchScript specifically
    because a converted quantized module doesn't reliably survive a plain
    pickle round-trip across processes -- so this tries TorchScript first,
    then a pickled module/state-dict."""
    try:
        model = torch.jit.load(model_path, map_location="cpu")
    except RuntimeError:
        obj = torch.load(model_path, map_location="cpu", weights_only=False)
        model = obj if isinstance(obj, torch.nn.Module) else ActivityCNN()
        if not isinstance(obj, torch.nn.Module):
            model.load_state_dict(obj)
    model.eval()
    return model


def build_track(subject_id: str, data_dir: str | Path, model: ActivityCNN, model_id: str = "full") -> list[dict]:
    subject = load_subject(data_dir, subject_id)
    globalized = globalize_subject(subject)
    return build_track_from_globalized(globalized, model, model_id=model_id, subject_id=subject_id)


def build_track_from_globalized(
    globalized, model: ActivityCNN, model_id: str = "full", subject_id: str = "?"
) -> list[dict]:
    """The same pipeline as `build_track`, taking an already-globalized
    `ats.resample.GlobalSamples` instead of loading+globalizing a subject
    itself. Used directly by `scripts/sweep.py` (docs/TASKS.md task 5A.3) so
    a degraded signal (`ats/degrade.py`) can be windowed and scored without
    a round trip through the raw archives for every sweep point."""
    all_windows = list(make_windows(globalized, window_s=WINDOW_LENGTH_S, hop_s=HOP_S))
    if not all_windows:
        return []

    windows: list = []
    tensors: list[np.ndarray] = []
    skipped = 0
    for w in all_windows:
        try:
            tensors.append(window_to_tensor(w))
            windows.append(w)
        except ValueError:
            # A whole channel (accelerometer or gyroscope) had zero real
            # samples for this window's burst -- happens for real data after
            # ats.ingest drops a corrupted-timestamp burst's channel
            # entirely (ats.ingest._sanitize_burst_rows). No gap-filling can
            # recover a channel with nothing to fill from, so this window is
            # skipped rather than fed a fabricated input.
            skipped += 1
    if skipped:
        print(f"[{subject_id}] skipped {skipped} window(s) with an entirely-missing channel")
    if not windows:
        return []

    probs = predict_probs(model, torch.from_numpy(np.stack(tensors, axis=0))).numpy()

    entries = []
    for i, (w, p) in enumerate(zip(windows, probs)):
        entry = {
            "window_id": f"w{i:06d}",
            "t_start": round(w.t_start, 3),
            "t_end": round(w.t_end, 3),
            "probs": [float(v) for v in p],
            "coverage": round(w.coverage, 4),
            "feature_summary": feature_summary(w),
            "model_id": model_id,
        }
        validate_window_track(entry)
        entries.append(entry)
    return entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ats.recognize",
        description="Emit a real-model window_track for one ExtraSensory subject "
        "(docs/TASKS.md task 2A.4) -- the trained-CNN counterpart to ats/oracle.py.",
    )
    parser.add_argument("--subject", required=True, help="Subject UUID.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory scripts/fetch_data.py staged archives under.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the trained model's state dict (.pt).")
    parser.add_argument("--model-id", default="full", help="model_id to stamp on every window_track entry.")
    parser.add_argument("--out", required=True, help="Path to write the window_track JSONL to.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model = load_model(args.model_path)
    entries = build_track(args.subject, args.data_dir, model, model_id=args.model_id)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"wrote {len(entries)} windows -> {out}")


if __name__ == "__main__":
    main()
