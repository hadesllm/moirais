"""Tests for nngrd."""
import numpy as np
import pytest
from morie.fn.nngrd import nngrd


def test_nngrd_basic():
    result = nngrd()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "NatNeighbor-Grid"


def test_nngrd_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = nngrd(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_nngrd_no_data():
    result = nngrd(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_nngrd_alias():
    from morie.fn.nngrd import nngrd
    assert nngrd is nngrd
