"""Tests for stkrgv."""
import numpy as np
import pytest
from moirais.fn.stkrgv import stkrgv


def test_stkrgv_basic():
    result = stkrgv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Kriging-Variance"


def test_stkrgv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stkrgv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stkrgv_no_data():
    result = stkrgv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stkrgv_alias():
    from moirais.fn.stkrgv import stkrgv
    assert stkrgv is stkrgv
