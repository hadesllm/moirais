"""Tests for vddln3."""
import numpy as np
import pytest
from morie.fn.vddln3 import vddln3


def test_vddln3_basic():
    result = vddln3()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Delaunay-3D"


def test_vddln3_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = vddln3(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_vddln3_no_data():
    result = vddln3(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_vddln3_alias():
    from morie.fn.vddln3 import vddln3
    assert vddln3 is vddln3
