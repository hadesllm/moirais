"""Tests for tmar."""
import numpy as np
import pytest
from moirais.fn.tmar import tmar


def test_tmar_basic():
    result = tmar()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TempTrend-AR"


def test_tmar_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tmar(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tmar_no_data():
    result = tmar(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tmar_alias():
    from moirais.fn.tmar import tmar
    assert tmar is tmar
