"""Tests for moirais.fn.sllrn — super learner."""

import numpy as np
import pytest

from moirais.fn.sllrn import sllrn


def test_basic_output():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 3))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.3
    result = sllrn(X, y, n_folds=3, seed=7)
    assert "weights" in result
    assert "cv_risks" in result


def test_weights_sum_to_one():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = X[:, 0] + rng.standard_normal(n)
    result = sllrn(X, y, n_folds=3, seed=1)
    assert np.sum(result["weights"]) == pytest.approx(1.0, abs=1e-4)


def test_ensemble_risk_le_worst():
    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 2))
    y = X[:, 0] + rng.standard_normal(n) * 0.5
    result = sllrn(X, y, n_folds=3, seed=42)
    assert result["ensemble_risk"] <= np.max(result["cv_risks"]) + 1e-6
