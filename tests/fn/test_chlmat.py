"""Tests for chlmat."""
import numpy as np
import pytest
from morie.fn.chlmat import chlmat


def test_chlmat_basic():
    result = chlmat()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Cholesky-Matern"


def test_chlmat_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlmat(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlmat_no_data():
    result = chlmat(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlmat_alias():
    from morie.fn.chlmat import chlmat
    assert chlmat is chlmat
