"""Tests for cdreal."""
import numpy as np
import pytest
from morie.fn.cdreal import cdreal


def test_cdreal_basic():
    result = cdreal()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ConditionalSim-Path"


def test_cdreal_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = cdreal(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_cdreal_no_data():
    result = cdreal(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_cdreal_alias():
    from morie.fn.cdreal import cdreal
    assert cdreal is cdreal
