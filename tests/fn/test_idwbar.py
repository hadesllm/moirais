"""Tests for idwbar."""
import numpy as np
import pytest
from morie.fn.idwbar import idwbar


def test_idwbar_basic():
    result = idwbar()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "IDW-Barrier"


def test_idwbar_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = idwbar(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_idwbar_no_data():
    result = idwbar(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_idwbar_alias():
    from morie.fn.idwbar import idwbar
    assert idwbar is idwbar
