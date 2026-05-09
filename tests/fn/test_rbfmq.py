"""Tests for rbfmq."""
import numpy as np
import pytest
from moirais.fn.rbfmq import rbfmq


def test_rbfmq_basic():
    result = rbfmq()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-Multiquadric"


def test_rbfmq_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfmq(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfmq_no_data():
    result = rbfmq(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfmq_alias():
    from moirais.fn.rbfmq import rbfmq
    assert rbfmq is rbfmq
