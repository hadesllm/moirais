"""Tests for moirais.fn.npsel -- Nonparametric model selection."""

import numpy as np
import pytest

from moirais.fn.npsel import npsel


@pytest.fixture()
def regression_data():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    y = np.sin(2 * np.pi * x) + rng.standard_normal(100) * 0.3
    return x, y


def test_returns_dict(regression_data):
    x, y = regression_data
    result = npsel(x, y)
    assert isinstance(result, dict)
    for k in ("best_bandwidth", "cv_scores", "bandwidths", "best_cv_score"):
        assert k in result


def test_best_bandwidth_positive(regression_data):
    x, y = regression_data
    result = npsel(x, y)
    assert result["best_bandwidth"] > 0


def test_cv_score_finite(regression_data):
    x, y = regression_data
    result = npsel(x, y)
    assert np.isfinite(result["best_cv_score"])


def test_custom_bandwidths(regression_data):
    x, y = regression_data
    bws = [0.05, 0.1, 0.2, 0.5]
    result = npsel(x, y, bandwidths=bws)
    assert result["best_bandwidth"] in bws


def test_epanechnikov_kernel(regression_data):
    x, y = regression_data
    result = npsel(x, y, kernel="epanechnikov")
    assert result["best_bandwidth"] > 0


def test_invalid_kernel(regression_data):
    x, y = regression_data
    with pytest.raises(ValueError, match="kernel"):
        npsel(x, y, kernel="triangle")


def test_invalid_nfolds(regression_data):
    x, y = regression_data
    with pytest.raises(ValueError, match="n_folds"):
        npsel(x, y, n_folds=1)


def test_dimension_mismatch():
    with pytest.raises(ValueError):
        npsel(np.array([1, 2]), np.array([1, 2, 3]))


def test_cheatsheet():
    from moirais.fn.npsel import cheatsheet
    assert "model selection" in cheatsheet().lower()
