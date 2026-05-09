"""Tests for tmbrk."""
import numpy as np
import pytest
from moirais.fn.tmbrk import tmbrk


def test_tmbrk_basic():
    result = tmbrk()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TempTrend-BreakDetect"


def test_tmbrk_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tmbrk(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tmbrk_no_data():
    result = tmbrk(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tmbrk_alias():
    from moirais.fn.tmbrk import tmbrk
    assert tmbrk is tmbrk
