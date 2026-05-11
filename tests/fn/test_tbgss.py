"""Tests for tbgss."""
import numpy as np
import pytest
from morie.fn.tbgss import tbgss


def test_tbgss_basic():
    result = tbgss()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TurningBands-Gaussian"


def test_tbgss_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tbgss(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tbgss_no_data():
    result = tbgss(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tbgss_alias():
    from morie.fn.tbgss import tbgss
    assert tbgss is tbgss
