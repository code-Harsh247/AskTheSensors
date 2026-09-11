"""Model compression (docs/TASKS.md task 5A.1, PRD Sec 6.3 extra credit):
two techniques, both pure post-training weight transforms with **no
training or fine-tuning step** -- deliberately, so nothing here needs a
Kaggle run.

- `quantize_int8`: post-training static int8 quantization (fuse Conv1d +
  BatchNorm1d + ReLU, calibrate on real windows, convert). Halves-to-quarters
  on-disk size and speeds up CPU inference; the *logical* parameter count is
  unchanged, only each weight's storage width shrinks.
- `prune_channels`: structured filter pruning. Keeps the highest-L1-norm
  output channels of each Conv1d layer and slices every weight that depends
  on that channel dimension (this layer's BatchNorm, the next layer's input
  channels, the final classifier's input dim) out of the trained `full`
  model -- a deterministic weight transfer into a genuinely smaller
  architecture (`ats.model.ActivityCNN(hidden_channels=...)`), not a mask
  applied to the same-size dense tensors (which would not shrink anything).

Distillation (training a smaller student model from scratch) is not
implemented here: unlike these two, it requires an actual training loop,
which per this project's convention runs on Kaggle, not locally. The
exit criterion (docs/TASKS.md Phase 5) asks for >=3 compressed configs, not
these three specific techniques by name; two pruning strengths plus
quantization already satisfy it without training.

Usage: python -m ats.compress --config quant8|pruned30|pruned60 [--base-model PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import convert, fuse_modules, get_default_qconfig, prepare

from ats.model import ActivityCNN, QuantizableActivityCNN, count_parameters

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = REPO_ROOT / "models" / "full" / "activity_cnn.pt"
DEFAULT_CALIB_DATA = REPO_ROOT / "results" / "raw" / "cnn_windows_val.npz"

# features.<i> indices of ActivityCNN's three Conv1d/BatchNorm1d/ReLU blocks
# (ats/model.py): [0,1,2], [4,5,6], [8,9,10] -- 3 and 7 are MaxPool1d, no
# parameters to fuse.
_FUSE_GROUPS = [["features.0", "features.1", "features.2"], ["features.4", "features.5", "features.6"], ["features.8", "features.9", "features.10"]]


def quantize_int8(model: ActivityCNN, calibration_batches: list[torch.Tensor]) -> nn.Module:
    """Post-training static int8 quantization. `calibration_batches` should
    be real (or at least realistic) window tensors -- static quantization
    picks activation scales from observed calibration-run statistics, not
    from the weights alone."""
    # Available quantized backends vary by platform/build -- e.g. this
    # project's Windows CPU torch build only ships "onednn", not the more
    # commonly-documented "fbgemm"/"qnnpack" -- so pick from whatever this
    # install actually supports rather than hardcoding one.
    supported = torch.backends.quantized.supported_engines
    backend = next((b for b in ("fbgemm", "qnnpack", "onednn") if b in supported), supported[0])
    torch.backends.quantized.engine = backend

    q_model = QuantizableActivityCNN()
    q_model.load_state_dict(model.state_dict())
    q_model.eval()

    q_model = fuse_modules(q_model, _FUSE_GROUPS)
    q_model.qconfig = get_default_qconfig(backend)
    prepare(q_model, inplace=True)
    with torch.no_grad():
        for batch in calibration_batches:
            q_model(batch)
    convert(q_model, inplace=True)
    q_model.eval()
    return q_model


def prune_channels(model: ActivityCNN, keep_fracs: tuple[float, float, float]) -> ActivityCNN:
    """Structured filter pruning: keep the `keep_fracs[i]` highest-L1-norm
    output channels of Conv1d layer i, slicing this layer's BatchNorm, the
    next layer's input-channel dimension, and (for the last layer) the
    classifier's input dimension to match. No fine-tuning -- a deterministic
    weight transfer, not a from-scratch or gradient-updated model."""
    convs = [model.features[0], model.features[4], model.features[8]]
    bns = [model.features[1], model.features[5], model.features[9]]
    old_h = tuple(c.out_channels for c in convs)
    new_h = tuple(max(1, round(h * f)) for h, f in zip(old_h, keep_fracs))

    keep_idx = []
    for conv, h_new in zip(convs, new_h):
        importance = conv.weight.detach().abs().sum(dim=(1, 2))
        idx = torch.argsort(importance, descending=True)[:h_new]
        idx, _ = torch.sort(idx)
        keep_idx.append(idx)

    pruned = ActivityCNN(hidden_channels=new_h)
    p_convs = [pruned.features[0], pruned.features[4], pruned.features[8]]
    p_bns = [pruned.features[1], pruned.features[5], pruned.features[9]]

    in_idx = None  # first conv's input is the fixed N_CHANNELS raw channels
    for conv, bn, p_conv, p_bn, idx in zip(convs, bns, p_convs, p_bns, keep_idx):
        w = conv.weight.detach()
        if in_idx is not None:
            w = w[:, in_idx, :]
        p_conv.weight.data.copy_(w[idx])
        if conv.bias is not None:
            p_conv.bias.data.copy_(conv.bias.detach()[idx])
        p_bn.weight.data.copy_(bn.weight.detach()[idx])
        p_bn.bias.data.copy_(bn.bias.detach()[idx])
        p_bn.running_mean.data.copy_(bn.running_mean.detach()[idx])
        p_bn.running_var.data.copy_(bn.running_var.detach()[idx])
        in_idx = idx

    pruned.classifier.weight.data.copy_(model.classifier.weight.detach()[:, keep_idx[-1]])
    pruned.classifier.bias.data.copy_(model.classifier.bias.detach())

    pruned.eval()
    return pruned


def save_config(model: nn.Module, out_dir: Path, params: int, torchscript: bool = False) -> Path:
    """Saves the whole module (not a state dict) -- pruning changes the
    architecture and a converted quantized module cannot be reconstructed
    from a bare state dict -- so `ats.recognize.load_model` loads whichever
    it finds (see that function's docstring).

    `torchscript=True` (used for quant8) scripts and saves via
    `torch.jit.save` instead of plain `torch.save`/pickle: a converted
    eager-mode quantized module's fused submodules (e.g. `ConvReLU1d`)
    don't reliably survive a raw pickle round-trip in a *different* process
    (`AttributeError: 'ConvReLU1d' object has no attribute '_modules'`,
    found by actually loading a saved quant8 checkpoint fresh, not by
    inspection) -- TorchScript is PyTorch's own supported path for
    persisting a quantized model.

    `params.json` records the logical parameter count, since it can't
    always be recovered by introspecting the saved module (a converted
    quantized layer's weights are packed, not plain `nn.Parameter`s)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "activity_cnn.pt"
    if torchscript:
        torch.jit.save(torch.jit.script(model), model_path)
    else:
        torch.save(model, model_path)
    (out_dir / "params.json").write_text(json.dumps({"params": params}), encoding="utf-8")
    return model_path


def _load_calibration_batches(calib_path: Path, n_batches: int, batch_size: int = 32) -> list[torch.Tensor]:
    data = np.load(calib_path)
    X = data["X"]
    rng = np.random.default_rng(0)
    n = min(len(X), n_batches * batch_size)
    idx = rng.choice(len(X), size=n, replace=False)
    sample = X[idx]
    return [torch.from_numpy(sample[i * batch_size : (i + 1) * batch_size]) for i in range(n_batches) if i * batch_size < n]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ats.compress",
        description="Produce one compressed model configuration (docs/TASKS.md task 5A.1).",
    )
    parser.add_argument("--config", required=True, choices=["quant8", "pruned30", "pruned60"])
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--calib-data", default=str(DEFAULT_CALIB_DATA), help="cnn_windows_val.npz, for quant8 calibration only.")
    parser.add_argument("--n-calib-batches", type=int, default=8)
    parser.add_argument("--out-dir", default=None, help="Defaults to models/<config>/.")
    args = parser.parse_args(argv)

    base = ActivityCNN()
    base.load_state_dict(torch.load(args.base_model, map_location="cpu"))
    base.eval()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "models" / args.config

    if args.config == "quant8":
        batches = _load_calibration_batches(Path(args.calib_data), args.n_calib_batches)
        model = quantize_int8(base, batches)
        params = count_parameters(base)  # quantization doesn't change the logical count
    else:
        pct_pruned = int(args.config.replace("pruned", ""))
        keep_frac = 1.0 - pct_pruned / 100.0
        model = prune_channels(base, (keep_frac, keep_frac, keep_frac))
        params = count_parameters(model)

    model_path = save_config(model, out_dir, params, torchscript=(args.config == "quant8"))
    print(f"wrote {model_path} ({model_path.stat().st_size / 1024:.1f} KB), params={params}")


if __name__ == "__main__":
    main()
