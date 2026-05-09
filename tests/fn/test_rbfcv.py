"""Tests for rbfcv."""
import numpy as np
import pytest
from moirais.fn.rbfcv import rbfcv


def test_rbfcv_basic():
    result = rbfcv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-CrossValidation"


def test_rbfcv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfcv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfcv_no_data():
    result = rbfcv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfcv_alias():
    from moirais.fn.rbfcv import rbfcv
    assert rbfcv is rbfcv
