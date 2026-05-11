"""Tests for tmkf."""
import numpy as np
import pytest
from morie.fn.tmkf import tmkf


def test_tmkf_basic():
    result = tmkf()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TempTrend-Kalman"


def test_tmkf_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tmkf(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tmkf_no_data():
    result = tmkf(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tmkf_alias():
    from morie.fn.tmkf import tmkf
    assert tmkf is tmkf
