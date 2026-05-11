"""Tests for tbreal."""
import numpy as np
import pytest
from morie.fn.tbreal import tbreal


def test_tbreal_basic():
    result = tbreal()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TurningBands-Realization"


def test_tbreal_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tbreal(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tbreal_no_data():
    result = tbreal(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tbreal_alias():
    from morie.fn.tbreal import tbreal
    assert tbreal is tbreal
