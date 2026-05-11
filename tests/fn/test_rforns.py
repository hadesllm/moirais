"""Tests for rforns."""
import numpy as np
import pytest
from morie.fn.rforns import rforns


def test_rforns_basic():
    result = rforns()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-OrnsteinUhlenbeck"


def test_rforns_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rforns(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rforns_no_data():
    result = rforns(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rforns_alias():
    from morie.fn.rforns import rforns
    assert rforns is rforns
