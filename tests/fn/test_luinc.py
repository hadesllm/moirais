"""Tests for luinc."""
import numpy as np
import pytest
from moirais.fn.luinc import luinc


def test_luinc_basic():
    result = luinc()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "LU-Incomplete"


def test_luinc_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = luinc(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_luinc_no_data():
    result = luinc(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_luinc_alias():
    from moirais.fn.luinc import luinc
    assert luinc is luinc
