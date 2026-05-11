"""Tests for rfms."""
import numpy as np
import pytest
from morie.fn.rfms import rfms


def test_rfms_basic():
    result = rfms()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-MultiScale"


def test_rfms_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfms(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfms_no_data():
    result = rfms(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfms_alias():
    from morie.fn.rfms import rfms
    assert rfms is rfms
