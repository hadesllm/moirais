"""Tests for rfsmt."""
import numpy as np
import pytest
from morie.fn.rfsmt import rfsmt


def test_rfsmt_basic():
    result = rfsmt()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-Smooth"


def test_rfsmt_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfsmt(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfsmt_no_data():
    result = rfsmt(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfsmt_alias():
    from morie.fn.rfsmt import rfsmt
    assert rfsmt is rfsmt
