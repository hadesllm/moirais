"""Tests for sbvar."""
import numpy as np
import pytest
from moirais.fn.sbvar import sbvar


def test_sbvar_basic():
    result = sbvar()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpatialBootstrap-Variance"


def test_sbvar_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sbvar(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sbvar_no_data():
    result = sbvar(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sbvar_alias():
    from moirais.fn.sbvar import sbvar
    assert sbvar is sbvar
