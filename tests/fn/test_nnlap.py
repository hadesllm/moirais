"""Tests for nnlap."""
import numpy as np
import pytest
from morie.fn.nnlap import nnlap


def test_nnlap_basic():
    result = nnlap()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "NatNeighbor-Laplace"


def test_nnlap_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = nnlap(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_nnlap_no_data():
    result = nnlap(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_nnlap_alias():
    from morie.fn.nnlap import nnlap
    assert nnlap is nnlap
