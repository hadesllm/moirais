"""Tests for luband."""
import numpy as np
import pytest
from moirais.fn.luband import luband


def test_luband_basic():
    result = luband()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "LU-Banded"


def test_luband_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = luband(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_luband_no_data():
    result = luband(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_luband_alias():
    from moirais.fn.luband import luband
    assert luband is luband
