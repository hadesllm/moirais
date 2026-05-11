"""Tests for sgreal."""
import numpy as np
import pytest
from morie.fn.sgreal import sgreal


def test_sgreal_basic():
    result = sgreal()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-Realization"


def test_sgreal_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sgreal(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sgreal_no_data():
    result = sgreal(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sgreal_alias():
    from morie.fn.sgreal import sgreal
    assert sgreal is sgreal
