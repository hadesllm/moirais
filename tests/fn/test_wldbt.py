"""Tests for morie.fn.wldbt -- Wild bootstrap for heteroskedastic regression."""

import numpy as np
import pytest

from morie.fn.wldbt import wldbt


@pytest.fixture()
def reg_data():
    rng = np.random.default_rng(42)
    n = 100
    x = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x])
    y = 2.0 + 1.5 * x + rng.standard_normal(n) * (1 + 0.5 * np.abs(x))
    return y, X


def test_returns_dict(reg_data):
    y, X = reg_data
    result = wldbt(y, X)
    assert isinstance(result, dict)
    for k in ("estimate", "se", "ci_lower", "ci_upper", "p_value", "n_boot"):
        assert k in result


def test_estimate_finite(reg_data):
    y, X = reg_data
    result = wldbt(y, X)
    assert np.isfinite(result["estimate"])
    assert np.isfinite(result["se"])
    assert result["se"] > 0


def test_ci_order(reg_data):
    y, X = reg_data
    result = wldbt(y, X)
    assert result["ci_lower"] <= result["ci_upper"]


def test_mammen_distribution(reg_data):
    y, X = reg_data
    result = wldbt(y, X, distribution="mammen")
    assert np.isfinite(result["estimate"])


def test_invalid_distribution(reg_data):
    y, X = reg_data
    with pytest.raises(ValueError, match="distribution"):
        wldbt(y, X, distribution="bad")


def test_dimension_mismatch():
    with pytest.raises(ValueError):
        wldbt(np.array([1, 2]), np.array([[1, 2, 3]]))


def test_coef_index_range():
    y = np.array([1.0, 2.0, 3.0])
    X = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
    with pytest.raises(ValueError, match="coef_index"):
        wldbt(y, X, coef_index=5)


def test_cheatsheet():
    from morie.fn.wldbt import cheatsheet
    assert "wild" in cheatsheet().lower()
