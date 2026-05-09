"""Tests for rfmax."""
import numpy as np
import pytest
from moirais.fn.rfmax import rfmax


def test_rfmax_basic():
    result = rfmax()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-MaxStable"


def test_rfmax_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfmax(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfmax_no_data():
    result = rfmax(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfmax_alias():
    from moirais.fn.rfmax import rfmax
    assert rfmax is rfmax
