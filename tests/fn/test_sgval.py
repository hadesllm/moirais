"""Tests for sgval."""
import numpy as np
import pytest
from morie.fn.sgval import sgval


def test_sgval_basic():
    result = sgval()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-CrossValidation"


def test_sgval_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sgval(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sgval_no_data():
    result = sgval(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sgval_alias():
    from morie.fn.sgval import sgval
    assert sgval is sgval
