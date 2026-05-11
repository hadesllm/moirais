"""Tests for rfwht."""
import numpy as np
import pytest
from morie.fn.rfwht import rfwht


def test_rfwht_basic():
    result = rfwht()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-WhiteNoise"


def test_rfwht_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfwht(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfwht_no_data():
    result = rfwht(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfwht_alias():
    from morie.fn.rfwht import rfwht
    assert rfwht is rfwht
