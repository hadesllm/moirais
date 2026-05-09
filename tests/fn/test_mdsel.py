"""Tests for moirais.fn.mdsel — model selection via CV loss."""

import numpy as np
import pytest

from moirais.fn.mdsel import mdsel


def test_basic_output():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 3))
    y = X[:, 0] + rng.standard_normal(n) * 0.5
    result = mdsel(X, y, n_folds=3, seed=7)
    assert "selected" in result
    assert "cv_risks" in result
    assert "model_names" in result


def test_ols_beats_intercept():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, 2))
    y = 2.0 * X[:, 0] + rng.standard_normal(n) * 0.3
    result = mdsel(X, y, n_folds=5, seed=1)
    ols_idx = result["model_names"].index("OLS")
    intercept_idx = result["model_names"].index("Intercept")
    assert result["cv_risks"][ols_idx] < result["cv_risks"][intercept_idx]


def test_1se_rule():
    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 2))
    y = X[:, 0] + rng.standard_normal(n)
    result = mdsel(X, y, n_folds=3, seed=42)
    assert 0 <= result["selected_1se"] < len(result["model_names"])


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        mdsel(np.array([]).reshape(0, 1), np.array([]))
