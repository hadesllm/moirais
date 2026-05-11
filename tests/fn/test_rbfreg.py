"""Tests for rbfreg."""
import numpy as np
import pytest
from morie.fn.rbfreg import rbfreg


def test_rbfreg_basic():
    result = rbfreg()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-Regularized"


def test_rbfreg_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfreg(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfreg_no_data():
    result = rbfreg(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfreg_alias():
    from morie.fn.rbfreg import rbfreg
    assert rbfreg is rbfreg
