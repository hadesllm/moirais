"""Tests for rncurv."""
import numpy as np
import pytest
from morie.fn.rncurv import rncurv


def test_rncurv_basic():
    result = rncurv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-Curvature"


def test_rncurv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rncurv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rncurv_no_data():
    result = rncurv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rncurv_alias():
    from morie.fn.rncurv import rncurv
    assert rncurv is rncurv
