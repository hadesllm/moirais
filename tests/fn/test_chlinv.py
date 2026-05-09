"""Tests for chlinv."""
import numpy as np
import pytest
from moirais.fn.chlinv import chlinv


def test_chlinv_basic():
    result = chlinv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "CholeskyInverse"


def test_chlinv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlinv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlinv_no_data():
    result = chlinv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlinv_alias():
    from moirais.fn.chlinv import chlinv
    assert chlinv is chlinv
