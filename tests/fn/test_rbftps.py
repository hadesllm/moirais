"""Tests for rbftps."""
import numpy as np
import pytest
from morie.fn.rbftps import rbftps


def test_rbftps_basic():
    result = rbftps()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-ThinPlate"


def test_rbftps_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbftps(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbftps_no_data():
    result = rbftps(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbftps_alias():
    from morie.fn.rbftps import rbftps
    assert rbftps is rbftps
