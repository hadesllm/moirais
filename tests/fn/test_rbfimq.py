"""Tests for rbfimq."""
import numpy as np
import pytest
from moirais.fn.rbfimq import rbfimq


def test_rbfimq_basic():
    result = rbfimq()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-InvMultiquadric"


def test_rbfimq_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfimq(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfimq_no_data():
    result = rbfimq(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfimq_alias():
    from moirais.fn.rbfimq import rbfimq
    assert rbfimq is rbfimq
