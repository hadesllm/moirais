"""Tests for tmloess."""
import numpy as np
import pytest
from moirais.fn.tmloess import tmloess


def test_tmloess_basic():
    result = tmloess()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TempTrend-LOESS"


def test_tmloess_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tmloess(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tmloess_no_data():
    result = tmloess(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tmloess_alias():
    from moirais.fn.tmloess import tmloess
    assert tmloess is tmloess
