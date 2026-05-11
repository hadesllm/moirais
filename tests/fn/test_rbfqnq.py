"""Tests for rbfqnq."""
import numpy as np
import pytest
from morie.fn.rbfqnq import rbfqnq


def test_rbfqnq_basic():
    result = rbfqnq()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-Quintic"


def test_rbfqnq_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfqnq(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfqnq_no_data():
    result = rbfqnq(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfqnq_alias():
    from morie.fn.rbfqnq import rbfqnq
    assert rbfqnq is rbfqnq
