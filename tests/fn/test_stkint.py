"""Tests for stkint."""
import numpy as np
import pytest
from morie.fn.stkint import stkint


def test_stkint_basic():
    result = stkint()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Kriging-TempInterp"


def test_stkint_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stkint(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stkint_no_data():
    result = stkint(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stkint_alias():
    from morie.fn.stkint import stkint
    assert stkint is stkint
