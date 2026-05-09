"""Tests for rndist."""
import numpy as np
import pytest
from moirais.fn.rndist import rndist


def test_rndist_basic():
    result = rndist()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-Distance"


def test_rndist_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rndist(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rndist_no_data():
    result = rndist(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rndist_alias():
    from moirais.fn.rndist import rndist
    assert rndist is rndist
