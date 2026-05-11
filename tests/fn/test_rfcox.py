"""Tests for rfcox."""
import numpy as np
import pytest
from morie.fn.rfcox import rfcox


def test_rfcox_basic():
    result = rfcox()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-CoxProcess"


def test_rfcox_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfcox(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfcox_no_data():
    result = rfcox(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfcox_alias():
    from morie.fn.rfcox import rfcox
    assert rfcox is rfcox
