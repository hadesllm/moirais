"""Tests for rndsgr."""
import numpy as np
import pytest
from morie.fn.rndsgr import rndsgr


def test_rndsgr_basic():
    result = rndsgr()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-Disaggregate"


def test_rndsgr_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rndsgr(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rndsgr_no_data():
    result = rndsgr(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rndsgr_alias():
    from morie.fn.rndsgr import rndsgr
    assert rndsgr is rndsgr
