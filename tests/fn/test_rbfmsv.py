"""Tests for rbfmsv."""
import numpy as np
import pytest
from moirais.fn.rbfmsv import rbfmsv


def test_rbfmsv_basic():
    result = rbfmsv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-MultiScale"


def test_rbfmsv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfmsv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfmsv_no_data():
    result = rbfmsv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfmsv_alias():
    from moirais.fn.rbfmsv import rbfmsv
    assert rbfmsv is rbfmsv
