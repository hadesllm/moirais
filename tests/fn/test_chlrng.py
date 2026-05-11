"""Tests for chlrng."""
import numpy as np
import pytest
from morie.fn.chlrng import chlrng


def test_chlrng_basic():
    result = chlrng()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "CholeskyRange"


def test_chlrng_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlrng(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlrng_no_data():
    result = chlrng(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlrng_alias():
    from morie.fn.chlrng import chlrng
    assert chlrng is chlrng
