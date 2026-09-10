"""Structural checks on ats/model.py's architecture -- shapes, parameter
count, and inference-time behavior. No training happens here or anywhere
locally (docs/TASKS.md 2A.2 training runs on Kaggle); this only verifies the
untrained architecture is wired correctly."""

from __future__ import annotations

import torch

from ats.contracts import CANONICAL_CLASSES
from ats.model import N_CHANNELS, N_CLASSES, ActivityCNN, count_parameters, predict_probs


def test_forward_pass_output_shape_matches_canonical_classes():
    model = ActivityCNN()
    x = torch.randn(4, N_CHANNELS, 101)  # batch of 4, the frozen window length in samples
    logits = model(x)
    assert logits.shape == (4, len(CANONICAL_CLASSES))
    assert N_CLASSES == len(CANONICAL_CLASSES)


def test_architecture_is_compact():
    model = ActivityCNN()
    n_params = count_parameters(model)
    # PRD Sec 6.3 rewards trading a *small* accuracy loss for a *large* cost
    # reduction -- only meaningful if the starting point is already small.
    assert n_params < 50_000, f"expected a compact backbone, got {n_params} parameters"


def test_forward_pass_is_agnostic_to_window_length():
    """AdaptiveAvgPool1d means the architecture doesn't hardcode the frozen
    101-sample window length -- a future window/hop change wouldn't require
    a new architecture."""
    model = ActivityCNN()
    for n_timesteps in (50, 101, 200):
        x = torch.randn(2, N_CHANNELS, n_timesteps)
        assert model(x).shape == (2, len(CANONICAL_CLASSES))


def test_predict_probs_is_a_valid_probability_distribution():
    model = ActivityCNN()
    x = torch.randn(3, N_CHANNELS, 101)
    probs = predict_probs(model, x)
    assert probs.shape == (3, len(CANONICAL_CLASSES))
    assert torch.all(probs >= 0)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(3), atol=1e-5)


def test_predict_probs_is_deterministic_in_eval_mode():
    model = ActivityCNN()
    x = torch.randn(2, N_CHANNELS, 101)
    first = predict_probs(model, x)
    second = predict_probs(model, x)
    assert torch.equal(first, second)
