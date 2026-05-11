"""Tests for lusim."""
import numpy as np
import pytest
from morie.fn.lusim import lusim


def test_lusim_basic():
    result = lusim()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "LU-Simulation"


def test_lusim_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = lusim(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_lusim_no_data():
    result = lusim(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_lusim_alias():
    from morie.fn.lusim import lusim
    assert lusim is lusim
