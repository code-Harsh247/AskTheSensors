"""Build the Phase 2 recognition-backbone datasets in one pass over the
cached ExtraSensory archives (docs/TASKS.md 2A.1, 2A.2):

  1. `results/raw/window_features.csv` -- the full engineered feature set
     per window (ats/features.py), used locally for class-imbalance
     analysis (2A.3) and EDA. Not needed by the CNN or its Kaggle training
     script.
  2. `results/raw/cnn_windows_<split>.npz` -- raw per-window 6-channel
     tensors (X: float32 [n_windows, 6, n_timesteps] in ats.model's
     INPUT_CHANNELS order; y: int32 canonical-class indices; subject_id;
     t_start), for upload to Kaggle to train ats/model.py's CNN and the
     required logistic-regression baseline (which needs only the raw
     window's magnitude mean/std -- cheaper to compute from these tensors
     on Kaggle than to also ship the full engineered-feature CSV there).

Both outputs apply the same coverage filter (see COVERAGE_THRESHOLD) so the
CNN and the two required baselines are compared on identical held-out
windows. Both are large, fully regenerable derived artifacts -- neither is
committed (results/raw/ is gitignored); re-run this script against the
cached archives to reproduce them byte-for-byte given the same code.

**Resumable by default.** Each subject's result is checkpointed to
`<out-dir>/_checkpoints/<split>/<subject_id>.{npz,csv,done}` as soon as that
subject finishes. Re-running the same command skips any subject whose
`.done` marker already exists, rather than reprocessing from scratch -- this
run is long enough (tens of minutes) that a killed session, a Kaggle stall,
or a Ctrl+C otherwise means losing everything. Pass `--restart` to wipe
checkpoints for the requested splits and reprocess everything. The final
`cnn_windows_<split>.npz` / `window_features.csv` are (re)built from
whatever checkpoints exist once all requested subjects are done (or already
were) -- this combine step is cheap and always reflects the latest state.

Requires `python scripts/fetch_data.py` to have staged the archives, and
`python scripts/make_splits.py` to have written splits/subject_splits.json.

`--data-dir` and `--out-dir` default to the local convention (`data/raw`,
`results/raw`) but can point anywhere -- e.g. on Kaggle, after cloning this
repo and uploading the three cached archives as a Dataset, point --data-dir
at wherever Kaggle mounts it and --out-dir at /kaggle/working. `--skip-csv`
omits the (large) engineered-feature CSV when only the CNN/baseline tensors
are needed -- the class counts for the imbalance write-up (2A.3) can be read
straight off the npz `y` arrays, so the CSV is optional there.

Usage: python scripts/build_feature_dataset.py [--splits train,val,test]
                                                [--data-dir PATH] [--out-dir PATH]
                                                [--skip-csv] [--restart]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from ats.contracts import CANONICAL_CLASSES  # noqa: E402
from ats.features import FEATURE_NAMES, compute_features  # noqa: E402
from ats.ingest import load_subject  # noqa: E402
from ats.model import INPUT_CHANNELS  # noqa: E402
from ats.resample import globalize_subject  # noqa: E402
from ats.splits import load_splits  # noqa: E402
from ats.windowing import HOP_S, WINDOW_LENGTH_S, Window, label_for_window, make_windows  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "raw"
SPLITS_PATH = REPO_ROOT / "splits" / "subject_splits.json"
CHECKPOINT_DIRNAME = "_checkpoints"

CSV_FIELDNAMES = ["subject_id", "split", "t_start", "t_end", "label", "coverage"] + list(FEATURE_NAMES)
LABEL_INDEX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

# Windows below this coverage are excluded from *both* outputs, so the CNN
# and the two required baselines are always compared on the same held-out
# windows. 0.95 keeps windows with only a sliver of gap-filled/missing
# signal (see ats/resample.py's interpolation policy) while dropping windows
# where a real gap ate a meaningful fraction of the window.
COVERAGE_THRESHOLD = 0.95

# (source, axis) per entry of ats.model.INPUT_CHANNELS, in that exact order.
_CHANNEL_SOURCE = [("acc", 0), ("acc", 1), ("acc", 2), ("gyro", 0), ("gyro", 1), ("gyro", 2)]


def _fill_gaps(values: list[float | None]) -> list[float]:
    """Forward-fill, then back-fill, any remaining None left by
    ats/resample.py's interpolation. Only ever applied to windows already
    above COVERAGE_THRESHOLD, so at most a handful of samples per window --
    a reasonable, simple choice for that little residual gap, and one a raw
    tensor (which can't hold None) requires either way."""
    filled = list(values)
    last = None
    for i, v in enumerate(filled):
        if v is None:
            filled[i] = last
        else:
            last = v
    first = next((v for v in filled if v is not None), None)
    if first is None:
        raise ValueError("window has no real samples at all; should have failed the coverage filter")
    for i, v in enumerate(filled):
        if v is None:
            filled[i] = first
        else:
            break
    return filled  # type: ignore[return-value]


def raw_tensor_for_window(w: Window) -> np.ndarray:
    """(len(INPUT_CHANNELS), n_timesteps) float32 array, gap-filled."""
    rows = []
    for source, axis in _CHANNEL_SOURCE:
        samples = w.acc if source == "acc" else w.gyro
        rows.append(_fill_gaps([sample[axis] for sample in samples]))
    return np.asarray(rows, dtype=np.float32)


def rows_for_subject(subject_id: str, split: str, data_dir: Path):
    """Yields (csv_row, label, tensor_or_None) for every window with usable
    ground truth. `tensor_or_None` is None for windows below
    COVERAGE_THRESHOLD -- still written to the CSV (labelled data is
    labelled data for EDA purposes) but excluded from the CNN/baseline
    dataset."""
    subject = load_subject(data_dir, subject_id)
    globalized = globalize_subject(subject)
    windows = make_windows(globalized, window_s=WINDOW_LENGTH_S, hop_s=HOP_S)
    for w in windows:
        label = label_for_window(w.t_start, w.t_end, globalized.label_spans)
        if label is None:
            continue
        csv_row = {
            "subject_id": subject_id,
            "split": split,
            "t_start": round(w.t_start, 3),
            "t_end": round(w.t_end, 3),
            "label": label,
            "coverage": round(w.coverage, 4),
        }
        csv_row.update(compute_features(w))

        tensor = None
        if w.coverage >= COVERAGE_THRESHOLD:
            tensor = raw_tensor_for_window(w)
        yield csv_row, label, tensor


def _checkpoint_dir(out_dir: Path, split: str) -> Path:
    return out_dir / CHECKPOINT_DIRNAME / split


def _empty_tensor_arrays():
    return (
        np.zeros((0, len(INPUT_CHANNELS), 0), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype="<U64"),
        np.zeros((0,), dtype=np.float64),
    )


def process_subject_checkpointed(subject_id: str, split: str, data_dir: Path, out_dir: Path, write_csv: bool) -> tuple[int, int, float]:
    """Process one subject and checkpoint its result. Returns (n_windows,
    n_tensors, elapsed_seconds). The `.done` marker is written last, so a
    process killed mid-write leaves no `.done` file and the subject is
    correctly reprocessed on the next run."""
    ckpt_dir = _checkpoint_dir(out_dir, split)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    npz_path = ckpt_dir / f"{subject_id}.npz"
    csv_path = ckpt_dir / f"{subject_id}.csv"
    done_path = ckpt_dir / f"{subject_id}.done"

    t0 = time.time()
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subject_ids: list[str] = []
    t_starts: list[float] = []
    csv_rows: list[dict] = []
    n_windows = n_tensors = 0

    for csv_row, label, tensor in rows_for_subject(subject_id, split, data_dir):
        n_windows += 1
        if write_csv:
            csv_rows.append(csv_row)
        if tensor is not None:
            xs.append(tensor)
            ys.append(LABEL_INDEX[label])
            subject_ids.append(subject_id)
            t_starts.append(csv_row["t_start"])
            n_tensors += 1

    if xs:
        X = np.stack(xs, axis=0)
        y = np.asarray(ys, dtype=np.int32)
        subj_arr = np.asarray(subject_ids)
        t_arr = np.asarray(t_starts, dtype=np.float64)
    else:
        X, y, subj_arr, t_arr = _empty_tensor_arrays()
    np.savez_compressed(npz_path, X=X, y=y, subject_id=subj_arr, t_start=t_arr)

    if write_csv:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(csv_rows)

    done_path.write_text("ok", encoding="ascii")
    return n_windows, n_tensors, time.time() - t0


def combine_split(out_dir: Path, split: str, write_csv: bool) -> int:
    """(Re)build cnn_windows_<split>.npz from every checkpointed subject for
    that split. Returns the total tensor count. Safe to call any time --
    reflects whatever checkpoints currently exist, whether from one
    continuous run or several resumed ones."""
    ckpt_dir = _checkpoint_dir(out_dir, split)
    done_subjects = sorted(p.stem for p in ckpt_dir.glob("*.done")) if ckpt_dir.is_dir() else []

    xs, ys, subs, ts = [], [], [], []
    for subject_id in done_subjects:
        data = np.load(ckpt_dir / f"{subject_id}.npz")
        if data["X"].shape[0] > 0:
            xs.append(data["X"])
            ys.append(data["y"])
            subs.append(data["subject_id"])
            ts.append(data["t_start"])

    if xs:
        X, y = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
        subject_id_arr, t_start_arr = np.concatenate(subs, axis=0), np.concatenate(ts, axis=0)
    else:
        X, y, subject_id_arr, t_start_arr = _empty_tensor_arrays()

    npz_path = out_dir / f"cnn_windows_{split}.npz"
    np.savez_compressed(
        npz_path,
        X=X, y=y, subject_id=subject_id_arr, t_start=t_start_arr,
        channels=np.asarray(INPUT_CHANNELS), classes=np.asarray(CANONICAL_CLASSES),
    )
    print(f"[{split}] combined {len(done_subjects)} subject(s), {len(y)} tensors -> {npz_path}")
    return len(y)


def combine_csv(out_dir: Path, wanted_splits: list[str]) -> int:
    """(Re)build the single window_features.csv across all requested splits
    from their checkpointed per-subject CSVs. Returns total row count."""
    out_path = out_dir / "window_features.csv"
    n_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for split in wanted_splits:
            ckpt_dir = _checkpoint_dir(out_dir, split)
            if not ckpt_dir.is_dir():
                continue
            for subject_id in sorted(p.stem for p in ckpt_dir.glob("*.done")):
                csv_path = ckpt_dir / f"{subject_id}.csv"
                if not csv_path.is_file():
                    continue
                with csv_path.open(newline="", encoding="utf-8") as in_f:
                    for row in csv.DictReader(in_f):
                        writer.writerow(row)
                        n_rows += 1
    print(f"combined {n_rows} CSV rows -> {out_path}")
    return n_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated subset of splits to build.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory fetch_data.py staged archives under.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory to write outputs under.")
    parser.add_argument("--skip-csv", action="store_true", help="Skip the (large) engineered-feature CSV; npz tensors only.")
    parser.add_argument(
        "--restart", action="store_true",
        help="Wipe checkpoints for the requested splits and reprocess every subject from scratch.",
    )
    args = parser.parse_args(argv)
    wanted_splits = [s.strip() for s in args.splits.split(",")]
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv = not args.skip_csv

    if args.restart:
        for split in wanted_splits:
            ckpt_dir = _checkpoint_dir(out_dir, split)
            if ckpt_dir.is_dir():
                shutil.rmtree(ckpt_dir)
                print(f"[{split}] --restart: wiped existing checkpoints")

    splits = load_splits(SPLITS_PATH)
    t0 = time.time()

    for split in wanted_splits:
        subjects = splits[split]
        for i, subject_id in enumerate(subjects, start=1):
            done_marker = _checkpoint_dir(out_dir, split) / f"{subject_id}.done"
            if done_marker.is_file():
                print(f"[{split} {i}/{len(subjects)}] {subject_id}: already done, skipping")
                continue
            n_windows, n_tensors, elapsed = process_subject_checkpointed(subject_id, split, data_dir, out_dir, write_csv)
            print(
                f"[{split} {i}/{len(subjects)}] {subject_id}: {n_windows} windows, "
                f"{n_tensors} above coverage {COVERAGE_THRESHOLD} ({elapsed:.1f}s)"
            )
        combine_split(out_dir, split, write_csv)

    if write_csv:
        combine_csv(out_dir, wanted_splits)

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
