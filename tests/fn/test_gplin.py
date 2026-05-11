"""Tests for gplin."""
import numpy as np
import pytest
from morie.fn.gplin import gplin


def test_gplin_basic():
    result = gplin()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GP-Linear"


def test_gplin_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = gplin(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_gplin_no_data():
    result = gplin(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_gplin_alias():
    from morie.fn.gplin import gplin
    assert gplin is gplin
