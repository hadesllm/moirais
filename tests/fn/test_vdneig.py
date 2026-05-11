"""Tests for vdneig."""
import numpy as np
import pytest
from morie.fn.vdneig import vdneig


def test_vdneig_basic():
    result = vdneig()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Voronoi-Neighbors"


def test_vdneig_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = vdneig(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_vdneig_no_data():
    result = vdneig(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_vdneig_alias():
    from morie.fn.vdneig import vdneig
    assert vdneig is vdneig
