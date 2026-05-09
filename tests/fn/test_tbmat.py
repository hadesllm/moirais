"""Tests for tbmat."""
import numpy as np
import pytest
from moirais.fn.tbmat import tbmat


def test_tbmat_basic():
    result = tbmat()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TurningBands-Matern"


def test_tbmat_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tbmat(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tbmat_no_data():
    result = tbmat(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tbmat_alias():
    from moirais.fn.tbmat import tbmat
    assert tbmat is tbmat
