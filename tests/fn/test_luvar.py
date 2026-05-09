"""Tests for luvar."""
import numpy as np
import pytest
from moirais.fn.luvar import luvar


def test_luvar_basic():
    result = luvar()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "LU-VarianceScaling"


def test_luvar_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = luvar(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_luvar_no_data():
    result = luvar(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_luvar_alias():
    from moirais.fn.luvar import luvar
    assert luvar is luvar
