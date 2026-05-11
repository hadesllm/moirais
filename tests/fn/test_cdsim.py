"""Tests for cdsim."""
import numpy as np
import pytest
from morie.fn.cdsim import cdsim


def test_cdsim_basic():
    result = cdsim()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ConditionalSim"


def test_cdsim_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = cdsim(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_cdsim_no_data():
    result = cdsim(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_cdsim_alias():
    from morie.fn.cdsim import cdsim
    assert cdsim is cdsim
