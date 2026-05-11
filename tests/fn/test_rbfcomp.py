"""Tests for rbfcomp."""
import numpy as np
import pytest
from morie.fn.rbfcomp import rbfcomp


def test_rbfcomp_basic():
    result = rbfcomp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-Wendland"


def test_rbfcomp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfcomp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfcomp_no_data():
    result = rbfcomp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfcomp_alias():
    from morie.fn.rbfcomp import rbfcomp
    assert rbfcomp is rbfcomp
