"""Tests for moirais.fn.mxkde — Mixture model KDE."""

import numpy as np
import pytest
from moirais.fn.mxkde import mxkde


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(-2, 0.5, 50), rng.normal(2, 0.5, 50)])
    result = mxkde(x, n_components=2, seed=42)
    assert isinstance(result, dict)
    for key in ("x_eval", "density", "weights", "means", "stds", "n_obs"):
        assert key in result


def test_weights_sum_to_one():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = mxkde(x, n_components=2, seed=42)
    assert abs(sum(result["weights"]) - 1.0) < 1e-6


def test_density_nonnegative():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = mxkde(x, n_components=2, seed=42)
    assert all(d >= -1e-10 for d in result["density"])


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 5"):
        mxkde(np.ones(3))
