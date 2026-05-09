"""Tests for lusim2."""
import numpy as np
import pytest
from moirais.fn.lusim2 import lusim2


def test_lusim2_basic():
    result = lusim2()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "LU-Sparse"


def test_lusim2_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = lusim2(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_lusim2_no_data():
    result = lusim2(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_lusim2_alias():
    from moirais.fn.lusim2 import lusim2
    assert lusim2 is lusim2
