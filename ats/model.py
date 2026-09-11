"""The Phase 2 recognition backbone (docs/TASKS.md task 2A.2): a compact
1D-CNN over the raw resampled 6-channel window, chosen over gradient
boosting on engineered features because it fits PRD Sec 6.3's edge-extra-
credit path (quantization, pruning, **and** knowledge distillation -- the
last of which has no natural analogue for a tree ensemble). See
docs/CITATIONS.md#torch--pytorch-python-library.

Kept deliberately small (~10k parameters): PRD Sec 6.3 rewards trading a
*small* accuracy loss for a *large* cost reduction, which only has room to
show up if the starting point is already small. Uses only Conv1d + BatchNorm
+ ReLU + pooling -- no exotic ops -- so it fuses and quantizes cleanly with
PyTorch's standard post-training quantization tooling in Phase 5.

Training happens on Kaggle (see scripts/train_cnn.py); this module is also
what loads trained weights back for local inference/profiling.
"""

from __future__ import annotations

import torch
from torch import nn

from ats.contracts import CANONICAL_CLASSES

N_CLASSES = len(CANONICAL_CLASSES)

# Fixed input channel order every caller (the Kaggle export/training script,
# and local inference) must agree on.
INPUT_CHANNELS: tuple[str, ...] = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
N_CHANNELS = len(INPUT_CHANNELS)


class ActivityCNN(nn.Module):
    """Conv1d(6->16) -> Conv1d(16->32) -> Conv1d(32->64) -> global average
    pool -> Linear(64->7). Input: (batch, N_CHANNELS, n_timesteps) of
    resampled, gap-free (already-imputed) sensor values; output: raw logits
    in CANONICAL_CLASSES order (apply softmax/argmax outside, e.g. via
    `predict_probs` below) -- not tied to a fixed window length, so a future
    change to WINDOW_LENGTH_S doesn't require a new architecture.

    `hidden_channels` defaults to the trained (16, 32, 64) widths; narrower
    values are used by `ats/compress.py`'s structured channel pruning
    (docs/TASKS.md task 5A.1) to build a smaller architecture that a pruned
    checkpoint's sliced weights actually fit into.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_classes: int = N_CLASSES,
        hidden_channels: tuple[int, int, int] = (16, 32, 64),
    ):
        super().__init__()
        h1, h2, h3 = hidden_channels
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, h1, kernel_size=5, padding=2),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(h1, h2, kernel_size=5, padding=2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(h2, h3, kernel_size=3, padding=1),
            nn.BatchNorm1d(h3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(h3, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class QuantizableActivityCNN(nn.Module):
    """Same architecture as ActivityCNN, with QuantStub/DeQuantStub added
    around it -- required by PyTorch's eager-mode static quantization API
    (`ats/compress.py`, docs/TASKS.md task 5A.1), not a different model.

    Deliberately kept in this module rather than in ats/compress.py: a
    class defined in a script that is itself run via `python -m X` gets
    pickled under a `__main__` identity that a *different* process running
    `python -m ats.profile` can't resolve (`torch.save`/`load` use pickle
    under the hood). ats/model.py is never run directly, only imported, so
    its classes' module identity is stable across every process that loads
    a saved quantized checkpoint.
    """

    def __init__(self, hidden_channels: tuple[int, int, int] = (16, 32, 64)):
        super().__init__()
        from torch.ao.quantization import DeQuantStub, QuantStub

        h1, h2, h3 = hidden_channels
        self.quant = QuantStub()
        self.features = nn.Sequential(
            nn.Conv1d(N_CHANNELS, h1, kernel_size=5, padding=2),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(h1, h2, kernel_size=5, padding=2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(h2, h3, kernel_size=3, padding=1),
            nn.BatchNorm1d(h3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(h3, N_CLASSES)
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return self.dequant(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def predict_probs(model: ActivityCNN, x: torch.Tensor) -> torch.Tensor:
    """Softmax probabilities in CANONICAL_CLASSES order, for filling a
    window_track entry's `probs` field. Puts the model in eval mode and
    disables gradients -- this is for inference, never training."""
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x), dim=-1)
