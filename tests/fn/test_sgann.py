"""Tests for sgann."""
import numpy as np
import pytest
from morie.fn.sgann import sgann


def test_sgann_basic():
    result = sgann()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-Annealing"


def test_sgann_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sgann(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sgann_no_data():
    result = sgann(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sgann_alias():
    from morie.fn.sgann import sgann
    assert sgann is sgann
