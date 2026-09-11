"""ats/compress.py (docs/TASKS.md task 5A.1): model compression with no
training step. `prune_channels` is checked against a fully hand-computed
fixture (every kept weight traced to its exact source value by
construction); `quantize_int8` is checked for the properties that actually
matter for the exit criterion -- it runs, produces valid probabilities, and
shrinks the on-disk size -- since its internals (calibrated int8 scales)
aren't hand-computable the way a deterministic weight slice is."""

from __future__ import annotations

import json

import torch

from ats.compress import prune_channels, quantize_int8, save_config
from ats.model import ActivityCNN, N_CHANNELS, count_parameters, predict_probs


def _fill_by_output_channel(weight: torch.Tensor) -> None:
    """weight[i, ...] = float(i + 1) for every element -- makes each output
    channel's L1 norm strictly increasing in i, so "keep the top-k by norm"
    has one unambiguous answer: the k highest indices."""
    for i in range(weight.shape[0]):
        weight.data[i].fill_(float(i + 1))


def _build_hand_worked_model() -> ActivityCNN:
    torch.manual_seed(0)
    model = ActivityCNN(hidden_channels=(4, 4, 4))
    convs = [model.features[0], model.features[4], model.features[8]]
    bns = [model.features[1], model.features[5], model.features[9]]
    for i, (conv, bn) in enumerate(zip(convs, bns)):
        _fill_by_output_channel(conv.weight)
        conv.bias.data.copy_(torch.tensor([1000.0, 2000.0, 3000.0, 4000.0]) + i * 10000)
        bn.weight.data.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0]))
        bn.bias.data.copy_(torch.tensor([100.0, 200.0, 300.0, 400.0]))
        bn.running_mean.data.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        bn.running_var.data.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    # classifier.weight[:, k] = float(k + 1) for every output class row.
    for k in range(4):
        model.classifier.weight.data[:, k].fill_(float(k + 1))
    model.classifier.bias.data.copy_(torch.tensor([9.0] * 7))
    model.eval()
    return model


def test_prune_channels_keeps_the_top_norm_channels_with_exact_weights():
    model = _build_hand_worked_model()

    pruned = prune_channels(model, keep_fracs=(0.5, 0.5, 0.5))

    # Every layer's construction makes channels 2 and 3 (values 3, 4) the
    # top-2 by L1 norm out of channels 0-3 (values 1, 2, 3, 4) -- kept in
    # ascending order [2, 3].
    convs = [pruned.features[0], pruned.features[4], pruned.features[8]]
    bns = [pruned.features[1], pruned.features[5], pruned.features[9]]
    for i, (conv, bn) in enumerate(zip(convs, bns)):
        assert conv.out_channels == 2
        assert torch.allclose(conv.weight, torch.tensor([3.0, 4.0]).view(2, 1, 1).expand_as(conv.weight))
        assert torch.equal(conv.bias, torch.tensor([3000.0, 4000.0]) + i * 10000)
        assert torch.equal(bn.weight, torch.tensor([30.0, 40.0]))
        assert torch.equal(bn.bias, torch.tensor([300.0, 400.0]))
        assert torch.equal(bn.running_mean, torch.tensor([3.0, 4.0]))
        assert torch.allclose(bn.running_var, torch.tensor([0.3, 0.4]))

    # First conv's in_channels is untouched (N_CHANNELS raw sensor axes);
    # conv2/conv3's in_channels shrink to match the previous layer's kept 2.
    assert convs[0].in_channels == N_CHANNELS
    assert convs[1].in_channels == 2
    assert convs[2].in_channels == 2

    # classifier.weight[:, k] = float(k+1) means kept columns 2,3 -> values 3,4.
    assert torch.allclose(pruned.classifier.weight[:, 0], torch.full((7,), 3.0))
    assert torch.allclose(pruned.classifier.weight[:, 1], torch.full((7,), 4.0))
    assert torch.equal(pruned.classifier.bias, model.classifier.bias)


def test_prune_channels_output_is_still_a_valid_probability_distribution():
    model = ActivityCNN()
    pruned = prune_channels(model, keep_fracs=(0.5, 0.5, 0.5))
    assert count_parameters(pruned) < count_parameters(model)

    x = torch.randn(3, N_CHANNELS, 101)
    probs = predict_probs(pruned, x)
    assert probs.shape == (3, 7)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_quantize_int8_runs_and_shrinks_disk_size(tmp_path):
    model = ActivityCNN()
    calibration = [torch.randn(8, N_CHANNELS, 101) for _ in range(4)]

    quantized = quantize_int8(model, calibration)

    x = torch.randn(3, N_CHANNELS, 101)
    with torch.no_grad():
        logits = quantized(x)
    assert logits.shape == (3, 7)
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-4)

    full_path = tmp_path / "full.pt"
    torch.save(model.state_dict(), full_path)
    quant_path = save_config(quantized, tmp_path / "quant_dir", params=count_parameters(model), torchscript=True)

    assert quant_path.stat().st_size < full_path.stat().st_size
    assert json.loads((tmp_path / "quant_dir" / "params.json").read_text())["params"] == count_parameters(model)

    # The actual bug this guards against only showed up on a genuine
    # save/load round trip (a converted quantized module's fused
    # submodules didn't survive plain pickling -- ats.recognize.load_model
    # is the real code path every config, including this one, is loaded
    # through in production).
    from ats.recognize import load_model

    reloaded = load_model(quant_path)
    reloaded_probs = predict_probs(reloaded, x)
    assert reloaded_probs.shape == (3, 7)
    assert torch.allclose(reloaded_probs.sum(dim=-1), torch.ones(3), atol=1e-4)


def test_save_config_writes_a_loadable_module_and_params_sidecar(tmp_path):
    model = ActivityCNN(hidden_channels=(8, 16, 32))
    path = save_config(model, tmp_path / "cfg", params=count_parameters(model))

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(loaded, ActivityCNN)
    x = torch.randn(2, N_CHANNELS, 101)
    assert predict_probs(loaded, x).shape == (2, 7)
