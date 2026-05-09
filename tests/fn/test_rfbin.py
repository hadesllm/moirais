"""Tests for rfbin."""
import numpy as np
import pytest
from moirais.fn.rfbin import rfbin


def test_rfbin_basic():
    result = rfbin()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-Binary"


def test_rfbin_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfbin(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfbin_no_data():
    result = rfbin(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfbin_alias():
    from moirais.fn.rfbin import rfbin
    assert rfbin is rfbin
