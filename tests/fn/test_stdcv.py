"""Tests for stdcv."""
import numpy as np
import pytest
from moirais.fn.stdcv import stdcv


def test_stdcv_basic():
    result = stdcv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Differentiability"


def test_stdcv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stdcv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stdcv_no_data():
    result = stdcv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stdcv_alias():
    from moirais.fn.stdcv import stdcv
    assert stdcv is stdcv
