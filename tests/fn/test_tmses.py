"""Tests for tmses."""
import numpy as np
import pytest
from morie.fn.tmses import tmses


def test_tmses_basic():
    result = tmses()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TempTrend-Seasonal"


def test_tmses_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tmses(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tmses_no_data():
    result = tmses(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tmses_alias():
    from morie.fn.tmses import tmses
    assert tmses is tmses
