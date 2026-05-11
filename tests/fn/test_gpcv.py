"""Tests for gpcv."""
import numpy as np
import pytest
from morie.fn.gpcv import gpcv


def test_gpcv_basic():
    result = gpcv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GP-CrossValidation"


def test_gpcv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = gpcv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_gpcv_no_data():
    result = gpcv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_gpcv_alias():
    from morie.fn.gpcv import gpcv
    assert gpcv is gpcv
