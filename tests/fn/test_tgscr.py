"""Tests for morie.fn.tgscr — Tangent space projection."""

import numpy as np
import pytest

from morie.fn.tgscr import tgscr


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 200
    score = rng.standard_normal(n) + 1.0
    nuisance = rng.standard_normal((n, 3))
    return score, nuisance


def test_returns_dict(synth):
    result = tgscr(*synth)
    assert isinstance(result, dict)
    for key in ("projection", "efficient_score", "info_bound", "info_loss_fraction", "n", "k"):
        assert key in result


def test_shapes(synth):
    score, nuisance = synth
    result = tgscr(score, nuisance)
    assert result["projection"].shape == score.shape
    assert result["efficient_score"].shape == score.shape


def test_orthogonality(synth):
    score, nuisance = synth
    result = tgscr(score, nuisance)
    for j in range(nuisance.shape[1]):
        dot = np.abs(np.mean(result["efficient_score"] * nuisance[:, j]))
        assert dot < 0.1, f"Efficient score not orthogonal to nuisance dim {j}"


def test_decomposition(synth):
    score, nuisance = synth
    result = tgscr(score, nuisance)
    reconstructed = result["projection"] + result["efficient_score"]
    np.testing.assert_allclose(reconstructed, score, atol=1e-10)


def test_info_loss_in_01(synth):
    result = tgscr(*synth)
    assert 0.0 <= result["info_loss_fraction"] <= 1.0


def test_1d_nuisance():
    rng = np.random.default_rng(7)
    score = rng.standard_normal(100)
    nuisance = rng.standard_normal(100)
    result = tgscr(score, nuisance)
    assert result["k"] == 1
    assert result["n"] == 100
