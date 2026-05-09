"""Tests for rbfgrd."""
import numpy as np
import pytest
from moirais.fn.rbfgrd import rbfgrd


def test_rbfgrd_basic():
    result = rbfgrd()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-Grid"


def test_rbfgrd_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfgrd(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfgrd_no_data():
    result = rbfgrd(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfgrd_alias():
    from moirais.fn.rbfgrd import rbfgrd
    assert rbfgrd is rbfgrd
