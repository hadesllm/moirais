"""Tests for morie.fn.emvbs -- Evidence maximization."""

import numpy as np
from morie.fn.emvbs import evidence_maximization


def test_returns_dict():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = X @ [1.0, 2.0, 0.0] + rng.standard_normal(50) * 0.5
    result = evidence_maximization(X, y)
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_posterior_mean_close():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 2))
    y = X @ [3.0, -1.0] + rng.standard_normal(100) * 0.3
    result = evidence_maximization(X, y)
    assert abs(result["posterior_mean"][0] - 3.0) < 1.0


def test_log_evidence_finite():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2))
    y = X @ [1.0, 1.0] + rng.standard_normal(50)
    result = evidence_maximization(X, y)
    assert np.isfinite(result["log_evidence"])
