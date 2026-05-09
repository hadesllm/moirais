"""Tests for moirais.fn.avgde — Average derivative estimation."""

import numpy as np
import pytest

from moirais.fn.avgde import avgde


@pytest.fixture()
def linear_data():
    rng = np.random.default_rng(42)
    n = 300
    x = rng.standard_normal(n)
    y = 2.5 * x + rng.normal(0, 0.5, n)
    return y, x


def test_returns_dict(linear_data):
    y, x = linear_data
    result = avgde(y, x)
    assert isinstance(result, dict)
    for key in ("avg_derivative", "se", "t_stat", "pval", "bandwidth", "n_obs"):
        assert key in result


def test_derivative_sign(linear_data):
    y, x = linear_data
    result = avgde(y, x)
    assert result["avg_derivative"] > 0


def test_se_positive(linear_data):
    y, x = linear_data
    result = avgde(y, x)
    assert result["se"] > 0


def test_multivariate():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.3, n)
    result = avgde(y, X)
    assert isinstance(result["avg_derivative"], list)
    assert len(result["avg_derivative"]) == 2


def test_mismatched_raises():
    with pytest.raises(ValueError, match="!="):
        avgde(np.ones(10), np.ones(5))


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 10"):
        avgde(np.ones(5), np.ones(5))


def test_pval_finite(linear_data):
    y, x = linear_data
    result = avgde(y, x)
    assert np.isfinite(result["pval"])
