"""Tests for morie.fn.cvtml — cross-validated TMLE."""

import numpy as np
import pytest

from morie.fn.cvtml import cvtml


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 2))
    T = (X[:, 0] + rng.standard_normal(n) > 0).astype(float)
    Y = 1.0 * T + X[:, 0] + rng.standard_normal(n) * 0.5
    return Y, T, X


def test_method_name(synth):
    Y, T, X = synth
    result = cvtml(Y, T, X, n_folds=3, seed=42)
    assert result["method"] == "CV-TMLE"


def test_ate_reasonable(synth):
    Y, T, X = synth
    result = cvtml(Y, T, X, n_folds=3, seed=42)
    assert abs(result["ate"] - 1.0) < 2.0


def test_fold_estimates_length(synth):
    Y, T, X = synth
    result = cvtml(Y, T, X, n_folds=5, seed=7)
    assert len(result["fold_estimates"]) == 5
