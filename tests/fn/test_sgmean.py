"""Tests for sgmean."""
import numpy as np
import pytest
from morie.fn.sgmean import sgmean


def test_sgmean_basic():
    result = sgmean()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-ExpectedValue"


def test_sgmean_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sgmean(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sgmean_no_data():
    result = sgmean(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sgmean_alias():
    from morie.fn.sgmean import sgmean
    assert sgmean is sgmean
