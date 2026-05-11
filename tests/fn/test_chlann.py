"""Tests for chlann."""
import numpy as np
import pytest
from morie.fn.chlann import chlann


def test_chlann_basic():
    result = chlann()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "CholeskyAnnealing"


def test_chlann_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlann(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlann_no_data():
    result = chlann(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlann_alias():
    from morie.fn.chlann import chlann
    assert chlann is chlann
