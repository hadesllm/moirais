"""Tests for nncvx."""
import numpy as np
import pytest
from morie.fn.nncvx import nncvx


def test_nncvx_basic():
    result = nncvx()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "NatNeighbor-ConvexHull"


def test_nncvx_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = nncvx(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_nncvx_no_data():
    result = nncvx(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_nncvx_alias():
    from morie.fn.nncvx import nncvx
    assert nncvx is nncvx
