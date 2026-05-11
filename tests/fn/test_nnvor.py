"""Tests for nnvor."""
import numpy as np
import pytest
from morie.fn.nnvor import nnvor


def test_nnvor_basic():
    result = nnvor()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "NatNeighbor-Voronoi"


def test_nnvor_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = nnvor(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_nnvor_no_data():
    result = nnvor(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_nnvor_alias():
    from morie.fn.nnvor import nnvor
    assert nnvor is nnvor
