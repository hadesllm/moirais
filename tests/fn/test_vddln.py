"""Tests for vddln."""
import numpy as np
import pytest
from morie.fn.vddln import vddln


def test_vddln_basic():
    result = vddln()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Delaunay-2D"


def test_vddln_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = vddln(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_vddln_no_data():
    result = vddln(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_vddln_alias():
    from morie.fn.vddln import vddln
    assert vddln is vddln
