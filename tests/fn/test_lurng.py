"""Tests for lurng."""
import numpy as np
import pytest
from morie.fn.lurng import lurng


def test_lurng_basic():
    result = lurng()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "LU-RangeDependent"


def test_lurng_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = lurng(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_lurng_no_data():
    result = lurng(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_lurng_alias():
    from morie.fn.lurng import lurng
    assert lurng is lurng
