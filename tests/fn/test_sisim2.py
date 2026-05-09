"""Tests for sisim2."""
import numpy as np
import pytest
from moirais.fn.sisim2 import sisim2


def test_sisim2_basic():
    result = sisim2()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SIS-MultiThreshold"


def test_sisim2_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sisim2(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sisim2_no_data():
    result = sisim2(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sisim2_alias():
    from moirais.fn.sisim2 import sisim2
    assert sisim2 is sisim2
