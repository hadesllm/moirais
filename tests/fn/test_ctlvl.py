"""Tests for ctlvl."""
import numpy as np
import pytest
from morie.fn.ctlvl import ctlvl


def test_ctlvl_basic():
    result = ctlvl()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Contour-Levels"


def test_ctlvl_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = ctlvl(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_ctlvl_no_data():
    result = ctlvl(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_ctlvl_alias():
    from morie.fn.ctlvl import ctlvl
    assert ctlvl is ctlvl
