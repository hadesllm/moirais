"""Tests for sgvar."""
import numpy as np
import pytest
from moirais.fn.sgvar import sgvar


def test_sgvar_basic():
    result = sgvar()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-Variance"


def test_sgvar_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sgvar(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sgvar_no_data():
    result = sgvar(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sgvar_alias():
    from moirais.fn.sgvar import sgvar
    assert sgvar is sgvar
