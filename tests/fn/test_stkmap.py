"""Tests for stkmap."""
import numpy as np
import pytest
from morie.fn.stkmap import stkmap


def test_stkmap_basic():
    result = stkmap()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Kriging-Map"


def test_stkmap_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stkmap(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stkmap_no_data():
    result = stkmap(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stkmap_alias():
    from morie.fn.stkmap import stkmap
    assert stkmap is stkmap
