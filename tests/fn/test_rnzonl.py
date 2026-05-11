"""Tests for rnzonl."""
import numpy as np
import pytest
from morie.fn.rnzonl import rnzonl


def test_rnzonl_basic():
    result = rnzonl()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-Zonal"


def test_rnzonl_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rnzonl(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rnzonl_no_data():
    result = rnzonl(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rnzonl_alias():
    from morie.fn.rnzonl import rnzonl
    assert rnzonl is rnzonl
