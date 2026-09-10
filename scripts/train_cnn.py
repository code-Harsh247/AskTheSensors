"""Train the Phase 2 recognition backbone -- the compact 1D-CNN (ats/model.py)
and the two required baselines (docs/TASKS.md 2A.2 exit criteria) -- on
Kaggle, not locally.

Intended to run **entirely on Kaggle**, right after
`scripts/build_feature_dataset.py` on the same machine (no local feature
building or training). Kaggle notebook setup:

    !git clone https://github.com/<your-fork>/AskTheSensors.git
    %cd AskTheSensors
    !pip install -e . -q
    !python scripts/fetch_data.py --out /kaggle/working/data/raw
    !python scripts/build_feature_dataset.py \\
        --data-dir /kaggle/working/data/raw --out-dir /kaggle/working/features --skip-csv
    !python scripts/train_cnn.py --data-dir /kaggle/working/features --out-dir /kaggle/working

Then download `/kaggle/working/activity_cnn.pt` and
`/kaggle/working/phase2_results.json` and place them at `models/full/`
locally. `phase2_results.json` includes the CNN's and both baselines'
macro-F1 (docs/TASKS.md exit criterion), the CNN's confusion matrix and
per-class P/R/F1 (Figure 2, via scripts/make_fig2.py), and the training
split's class counts (for the 2A.3 imbalance write-up).

Reuses ats.model (the architecture), ats.imbalance (class weights), and
ats.eval.metrics (macro-F1, confusion matrix, per-class P/R/F1) -- the exact
same, already hand-tested implementations the rest of the project uses, so
this isn't a second, potentially-diverging reimplementation of those
metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from ats.contracts import CANONICAL_CLASSES  # noqa: E402
from ats.eval import metrics  # noqa: E402
from ats.imbalance import class_counts, class_weights  # noqa: E402
from ats.model import ActivityCNN  # noqa: E402

N_CLASSES = len(CANONICAL_CLASSES)
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SEED = 0


def load_split(data_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_dir / f"cnn_windows_{name}.npz", allow_pickle=True)
    return data["X"], data["y"]


def weight_tensor(y_train: np.ndarray) -> torch.Tensor:
    labels = [CANONICAL_CLASSES[i] for i in y_train]
    weights = class_weights(labels, CANONICAL_CLASSES)
    return torch.tensor([weights[c] for c in CANONICAL_CLASSES], dtype=torch.float32)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """macro-F1, per-class P/R/F1, and the confusion matrix, all via
    ats.eval.metrics (the same implementation B's stack and tests use)."""
    true_labels = [CANONICAL_CLASSES[i] for i in y_true]
    pred_labels = [CANONICAL_CLASSES[i] for i in y_pred]
    return {
        "macro_f1": metrics.macro_f1(pred_labels, true_labels, CANONICAL_CLASSES),
        "per_class": metrics.per_class_prf(pred_labels, true_labels, CANONICAL_CLASSES),
        "confusion_matrix": metrics.confusion_matrix(pred_labels, true_labels, CANONICAL_CLASSES),
    }


def majority_class_baseline(y_train: np.ndarray, y_val: np.ndarray) -> dict:
    majority = int(np.bincount(y_train, minlength=N_CLASSES).argmax())
    preds = np.full_like(y_val, majority)
    result = score(y_val, preds)
    result["majority_class"] = CANONICAL_CLASSES[majority]
    return result


def logistic_regression_baseline(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """"Logistic regression on per-window mean/std" (docs/TASKS.md Phase 2
    exit criteria, verbatim) -- computed from the acc-magnitude channel of
    the raw window tensor (channels 0-2 are acc_x/y/z; see
    ats.model.INPUT_CHANNELS), not the full engineered feature set. A tiny,
    literal baseline by design, not a second feature-rich model."""

    def magnitude_mean_std(X: np.ndarray) -> np.ndarray:
        mag = np.sqrt((X[:, 0, :] ** 2 + X[:, 1, :] ** 2 + X[:, 2, :] ** 2))
        return np.stack([mag.mean(axis=1), mag.std(axis=1)], axis=1)

    feats_train = torch.tensor(magnitude_mean_std(X_train), dtype=torch.float32)
    feats_val = torch.tensor(magnitude_mean_std(X_val), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    model = nn.Linear(2, N_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor(y_train))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(500):
        optimizer.zero_grad()
        criterion(model(feats_train), y_train_t).backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(feats_val).argmax(dim=1).numpy()
    return score(y_val, preds)


def train_cnn(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple[ActivityCNN, dict]:
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}")

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    X_val_t = torch.tensor(X_val).to(device)

    model = ActivityCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor(y_train).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).argmax(dim=1).cpu().numpy()
        f1 = metrics.macro_f1([CANONICAL_CLASSES[i] for i in val_preds], [CANONICAL_CLASSES[i] for i in y_val], CANONICAL_CLASSES)
        print(f"epoch {epoch:2d}/{EPOCHS}  val macro-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t.to(device)).argmax(dim=1).cpu().numpy()
    result = score(y_val, val_preds)
    result["best_epoch_f1"] = best_f1
    return model, result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="results/raw", help="Directory with cnn_windows_{train,val,test}.npz.")
    parser.add_argument("--out-dir", default=".", help="Directory to write activity_cnn.pt and phase2_results.json to.")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(data_dir, "train")
    X_val, y_val = load_split(data_dir, "val")
    print(f"train: {X_train.shape}, val: {X_val.shape}")

    results: dict = {"train_class_counts": class_counts([CANONICAL_CLASSES[i] for i in y_train], CANONICAL_CLASSES)}

    print("\n=== Baseline (a): majority class ===")
    results["majority_class"] = majority_class_baseline(y_train, y_val)
    print(f"macro-F1 = {results['majority_class']['macro_f1']:.4f}")

    print("\n=== Baseline (b): logistic regression on per-window mean/std ===")
    results["logistic_regression"] = logistic_regression_baseline(X_train, y_train, X_val, y_val)
    print(f"macro-F1 = {results['logistic_regression']['macro_f1']:.4f}")

    print("\n=== ats/model.py: ActivityCNN ===")
    model, cnn_result = train_cnn(X_train, y_train, X_val, y_val)
    results["cnn"] = cnn_result
    print(f"macro-F1 = {cnn_result['macro_f1']:.4f}")

    torch.save(model.state_dict(), out_dir / "activity_cnn.pt")
    with (out_dir / "phase2_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Exit criterion (docs/TASKS.md Phase 2, Member A) ===")
    cnn_f1 = results["cnn"]["macro_f1"]
    maj_f1 = results["majority_class"]["macro_f1"]
    lr_f1 = results["logistic_regression"]["macro_f1"]
    print(f"CNN macro-F1:                       {cnn_f1:.4f}")
    print(f"beats majority-class baseline:      {cnn_f1 > maj_f1} (margin {cnn_f1 - maj_f1:+.4f})")
    print(f"beats logistic-regression baseline: {cnn_f1 > lr_f1} (margin {cnn_f1 - lr_f1:+.4f})")
    print(f"\nSaved {out_dir / 'activity_cnn.pt'} and {out_dir / 'phase2_results.json'}")
    print("Download both from Kaggle's output panel; place them at models/full/ locally.")


if __name__ == "__main__":
    main()
