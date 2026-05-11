"""Tests for idwbst."""
import numpy as np
import pytest
from morie.fn.idwbst import idwbst


def test_idwbst_basic():
    result = idwbst()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "IDW-Boosted"


def test_idwbst_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = idwbst(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_idwbst_no_data():
    result = idwbst(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_idwbst_alias():
    from morie.fn.idwbst import idwbst
    assert idwbst is idwbst
