"""Tests for stch."""
import numpy as np
import pytest
from morie.fn.stch import stch


def test_stch_basic():
    result = stch()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-CressieHuang"


def test_stch_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stch(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stch_no_data():
    result = stch(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stch_alias():
    from morie.fn.stch import stch
    assert stch is stch
