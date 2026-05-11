"""Tests for siconn."""
import numpy as np
import pytest
from morie.fn.siconn import siconn


def test_siconn_basic():
    result = siconn()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SIS-Connectivity"


def test_siconn_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = siconn(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_siconn_no_data():
    result = siconn(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_siconn_alias():
    from morie.fn.siconn import siconn
    assert siconn is siconn
