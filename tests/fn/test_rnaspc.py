"""Tests for rnaspc."""
import numpy as np
import pytest
from morie.fn.rnaspc import rnaspc


def test_rnaspc_basic():
    result = rnaspc()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-Aspect"


def test_rnaspc_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rnaspc(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rnaspc_no_data():
    result = rnaspc(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rnaspc_alias():
    from morie.fn.rnaspc import rnaspc
    assert rnaspc is rnaspc
