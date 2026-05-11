"""Tests for tbsph."""
import numpy as np
import pytest
from morie.fn.tbsph import tbsph


def test_tbsph_basic():
    result = tbsph()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TurningBands-Spherical"


def test_tbsph_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tbsph(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tbsph_no_data():
    result = tbsph(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tbsph_alias():
    from morie.fn.tbsph import tbsph
    assert tbsph is tbsph
