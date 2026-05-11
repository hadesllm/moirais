"""Tests for rnwatr."""
import numpy as np
import pytest
from morie.fn.rnwatr import rnwatr


def test_rnwatr_basic():
    result = rnwatr()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-Watershed"


def test_rnwatr_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rnwatr(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rnwatr_no_data():
    result = rnwatr(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rnwatr_alias():
    from morie.fn.rnwatr import rnwatr
    assert rnwatr is rnwatr
