"""Tests for rbfgss."""
import numpy as np
import pytest
from morie.fn.rbfgss import rbfgss


def test_rbfgss_basic():
    result = rbfgss()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-Gaussian"


def test_rbfgss_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfgss(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfgss_no_data():
    result = rbfgss(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfgss_alias():
    from morie.fn.rbfgss import rbfgss
    assert rbfgss is rbfgss
