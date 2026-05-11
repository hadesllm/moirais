"""Tests for gpwarp."""
import numpy as np
import pytest
from morie.fn.gpwarp import gpwarp


def test_gpwarp_basic():
    result = gpwarp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GP-InputWarping"


def test_gpwarp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = gpwarp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_gpwarp_no_data():
    result = gpwarp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_gpwarp_alias():
    from morie.fn.gpwarp import gpwarp
    assert gpwarp is gpwarp
