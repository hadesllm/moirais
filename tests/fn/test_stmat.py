"""Tests for stmat."""
import numpy as np
import pytest
from morie.fn.stmat import stmat


def test_stmat_basic():
    result = stmat()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Matern"


def test_stmat_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stmat(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stmat_no_data():
    result = stmat(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stmat_alias():
    from morie.fn.stmat import stmat
    assert stmat is stmat
