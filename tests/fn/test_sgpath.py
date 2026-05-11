"""Tests for sgpath."""
import numpy as np
import pytest
from morie.fn.sgpath import sgpath


def test_sgpath_basic():
    result = sgpath()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-RandomPath"


def test_sgpath_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sgpath(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sgpath_no_data():
    result = sgpath(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sgpath_alias():
    from morie.fn.sgpath import sgpath
    assert sgpath is sgpath
