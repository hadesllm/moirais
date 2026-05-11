"""Tests for rfcat."""
import numpy as np
import pytest
from morie.fn.rfcat import rfcat


def test_rfcat_basic():
    result = rfcat()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-Categorical"


def test_rfcat_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfcat(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfcat_no_data():
    result = rfcat(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfcat_alias():
    from morie.fn.rfcat import rfcat
    assert rfcat is rfcat
