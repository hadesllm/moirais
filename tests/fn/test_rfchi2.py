"""Tests for rfchi2."""
import numpy as np
import pytest
from morie.fn.rfchi2 import rfchi2


def test_rfchi2_basic():
    result = rfchi2()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-ChiSquared"


def test_rfchi2_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfchi2(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfchi2_no_data():
    result = rfchi2(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfchi2_alias():
    from morie.fn.rfchi2 import rfchi2
    assert rfchi2 is rfchi2
