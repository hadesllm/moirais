"""Tests for rbfcub."""
import numpy as np
import pytest
from moirais.fn.rbfcub import rbfcub


def test_rbfcub_basic():
    result = rbfcub()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-Cubic"


def test_rbfcub_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfcub(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfcub_no_data():
    result = rbfcub(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfcub_alias():
    from moirais.fn.rbfcub import rbfcub
    assert rbfcub is rbfcub
