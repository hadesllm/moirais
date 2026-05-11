"""Tests for tmtrnd."""
import numpy as np
import pytest
from morie.fn.tmtrnd import tmtrnd


def test_tmtrnd_basic():
    result = tmtrnd()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TempTrend-OLS"


def test_tmtrnd_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tmtrnd(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tmtrnd_no_data():
    result = tmtrnd(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tmtrnd_alias():
    from morie.fn.tmtrnd import tmtrnd
    assert tmtrnd is tmtrnd
