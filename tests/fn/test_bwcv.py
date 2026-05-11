"""Tests for morie.fn.bwcv — Bandwidth selection via LOO cross-validation."""

import numpy as np
import pytest

from morie.fn.bwcv import bwcv


@pytest.fixture()
def reg_data():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 80)
    y = np.sin(x) + rng.normal(0, 0.3, 80)
    return x, y


def test_returns_dict(reg_data):
    x, y = reg_data
    result = bwcv(x, y)
    assert isinstance(result, dict)
    for key in ("bandwidth", "cv_score", "method", "kernel", "n_obs"):
        assert key in result


def test_bandwidth_positive(reg_data):
    x, y = reg_data
    result = bwcv(x, y)
    assert result["bandwidth"] > 0


def test_cv_score_finite(reg_data):
    x, y = reg_data
    result = bwcv(x, y)
    assert np.isfinite(result["cv_score"])


def test_density_method():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(80)
    y = np.zeros_like(x)
    result = bwcv(x, y, method="density")
    assert result["method"] == "density"
    assert result["bandwidth"] > 0


def test_epanechnikov_kernel(reg_data):
    x, y = reg_data
    result = bwcv(x, y, kernel="epanechnikov")
    assert result["kernel"] == "epanechnikov"


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 4"):
        bwcv(np.array([1, 2, 3]), np.array([1, 2, 3]))


def test_unknown_method_raises():
    x = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="Unknown method"):
        bwcv(x, x, method="bayesian")


def test_cv_bandwidth_reasonable(reg_data):
    x, y = reg_data
    result = bwcv(x, y)
    assert 0.01 < result["bandwidth"] < 50.0
