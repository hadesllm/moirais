"""Tests for moirais.fn.advi -- ADVI mean-field."""

import numpy as np
from moirais.fn.advi import advi_meanfield


def test_returns_dict():
    result = advi_meanfield(lambda x: -0.5 * float(x @ x), dim=1, n_iter=50)
    assert isinstance(result, dict)
    assert "variational_mean" in result
    assert "variational_std" in result


def test_elbo_history_length():
    result = advi_meanfield(lambda x: -0.5 * float(x @ x), dim=1, n_iter=100)
    assert len(result["elbo_history"]) == 100


def test_finds_standard_normal():
    result = advi_meanfield(
        lambda x: -0.5 * float(x @ x), dim=1,
        n_iter=2000, learning_rate=0.05, n_samples=20, seed=42
    )
    assert abs(result["variational_mean"][0]) < 1.0
    assert result["variational_std"][0] > 0.1


def test_invalid_dim():
    try:
        advi_meanfield(lambda x: 0.0, dim=0)
        assert False
    except ValueError:
        pass


def test_multivariate():
    def log_target(x):
        return -0.5 * float(x @ x)
    result = advi_meanfield(log_target, dim=3, n_iter=500, seed=42)
    assert len(result["variational_mean"]) == 3
    assert len(result["variational_std"]) == 3
