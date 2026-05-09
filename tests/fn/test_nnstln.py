"""Tests for nnstln."""
import numpy as np
import pytest
from moirais.fn.nnstln import nnstln


def test_nnstln_basic():
    result = nnstln()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "NatNeighbor-Gradient"


def test_nnstln_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = nnstln(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_nnstln_no_data():
    result = nnstln(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_nnstln_alias():
    from moirais.fn.nnstln import nnstln
    assert nnstln is nnstln
