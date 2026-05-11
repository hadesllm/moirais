"""Tests for tmexp."""
import numpy as np
import pytest
from morie.fn.tmexp import tmexp


def test_tmexp_basic():
    result = tmexp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TempTrend-ExpSmooth"


def test_tmexp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tmexp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tmexp_no_data():
    result = tmexp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tmexp_alias():
    from morie.fn.tmexp import tmexp
    assert tmexp is tmexp
