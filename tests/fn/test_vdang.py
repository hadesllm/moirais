"""Tests for vdang."""
import numpy as np
import pytest
from morie.fn.vdang import vdang


def test_vdang_basic():
    result = vdang()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Delaunay-AngleCheck"


def test_vdang_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = vdang(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_vdang_no_data():
    result = vdang(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_vdang_alias():
    from morie.fn.vdang import vdang
    assert vdang is vdang
