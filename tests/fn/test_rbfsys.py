"""Tests for rbfsys."""
import numpy as np
import pytest
from morie.fn.rbfsys import rbfsys


def test_rbfsys_basic():
    result = rbfsys()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-SystemSolve"


def test_rbfsys_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfsys(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfsys_no_data():
    result = rbfsys(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfsys_alias():
    from morie.fn.rbfsys import rbfsys
    assert rbfsys is rbfsys
