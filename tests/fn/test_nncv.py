"""Tests for nncv."""
import numpy as np
import pytest
from moirais.fn.nncv import nncv


def test_nncv_basic():
    result = nncv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "NatNeighbor-CV"


def test_nncv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = nncv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_nncv_no_data():
    result = nncv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_nncv_alias():
    from moirais.fn.nncv import nncv
    assert nncv is nncv
