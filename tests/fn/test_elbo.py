"""Tests for moirais.fn.elbo -- ELBO computation."""

import numpy as np
from moirais.fn.elbo import compute_elbo


def test_returns_dict():
    result = compute_elbo(
        lambda x: -0.5 * float(x @ x), [0.0], [1.0], n_samples=100
    )
    assert isinstance(result, dict)
    assert "elbo" in result
    assert "entropy" in result


def test_entropy_positive():
    result = compute_elbo(
        lambda x: -0.5 * float(x @ x), [0.0], [1.0], n_samples=100
    )
    assert result["entropy"] > 0


def test_elbo_standard_normal():
    result = compute_elbo(
        lambda x: -0.5 * float(x @ x), [0.0], [1.0], n_samples=5000, seed=42
    )
    entropy = 0.5 * (1 + np.log(2 * np.pi))
    expected_log_target = -0.5
    expected_elbo = expected_log_target + entropy
    assert abs(result["elbo"] - expected_elbo) < 0.5


def test_se_positive():
    result = compute_elbo(
        lambda x: -0.5 * float(x @ x), [0.0], [1.0], n_samples=100
    )
    assert result["elbo_se"] > 0


def test_invalid_std():
    try:
        compute_elbo(lambda x: 0.0, [0.0], [0.0])
        assert False
    except ValueError:
        pass


def test_mismatched_dims():
    try:
        compute_elbo(lambda x: 0.0, [0.0, 0.0], [1.0])
        assert False
    except ValueError:
        pass
