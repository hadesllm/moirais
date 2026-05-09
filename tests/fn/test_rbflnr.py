"""Tests for rbflnr."""
import numpy as np
import pytest
from moirais.fn.rbflnr import rbflnr


def test_rbflnr_basic():
    result = rbflnr()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-Linear"


def test_rbflnr_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbflnr(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbflnr_no_data():
    result = rbflnr(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbflnr_alias():
    from moirais.fn.rbflnr import rbflnr
    assert rbflnr is rbflnr
