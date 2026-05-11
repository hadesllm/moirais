"""Tests for sbstt."""
import numpy as np
import pytest
from morie.fn.sbstt import sbstt


def test_sbstt_basic():
    result = sbstt()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpatialBootstrap-Stationary"


def test_sbstt_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sbstt(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sbstt_no_data():
    result = sbstt(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sbstt_alias():
    from morie.fn.sbstt import sbstt
    assert sbstt is sbstt
