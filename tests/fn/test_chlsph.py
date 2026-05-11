"""Tests for chlsph."""
import numpy as np
import pytest
from morie.fn.chlsph import chlsph


def test_chlsph_basic():
    result = chlsph()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Cholesky-Spherical"


def test_chlsph_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlsph(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlsph_no_data():
    result = chlsph(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlsph_alias():
    from morie.fn.chlsph import chlsph
    assert chlsph is chlsph
