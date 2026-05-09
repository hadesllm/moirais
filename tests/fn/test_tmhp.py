"""Tests for tmhp."""
import numpy as np
import pytest
from moirais.fn.tmhp import tmhp


def test_tmhp_basic():
    result = tmhp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TempTrend-HP"


def test_tmhp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tmhp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tmhp_no_data():
    result = tmhp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tmhp_alias():
    from moirais.fn.tmhp import tmhp
    assert tmhp is tmhp
