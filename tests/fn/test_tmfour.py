"""Tests for tmfour."""
import numpy as np
import pytest
from moirais.fn.tmfour import tmfour


def test_tmfour_basic():
    result = tmfour()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TempTrend-Fourier"


def test_tmfour_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tmfour(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tmfour_no_data():
    result = tmfour(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tmfour_alias():
    from moirais.fn.tmfour import tmfour
    assert tmfour is tmfour
