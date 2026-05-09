"""Tests for rncost."""
import numpy as np
import pytest
from moirais.fn.rncost import rncost


def test_rncost_basic():
    result = rncost()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-CostSurface"


def test_rncost_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rncost(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rncost_no_data():
    result = rncost(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rncost_alias():
    from moirais.fn.rncost import rncost
    assert rncost is rncost
