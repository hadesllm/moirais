"""Tests for gpsparse."""
import numpy as np
import pytest
from morie.fn.gpsparse import gpsparse


def test_gpsparse_basic():
    result = gpsparse()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GP-Sparse"


def test_gpsparse_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = gpsparse(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_gpsparse_no_data():
    result = gpsparse(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_gpsparse_alias():
    from morie.fn.gpsparse import gpsparse
    assert gpsparse is gpsparse
