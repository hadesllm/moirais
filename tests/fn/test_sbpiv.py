"""Tests for sbpiv."""
import numpy as np
import pytest
from morie.fn.sbpiv import sbpiv


def test_sbpiv_basic():
    result = sbpiv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpatialBootstrap-Pivotal"


def test_sbpiv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sbpiv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sbpiv_no_data():
    result = sbpiv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sbpiv_alias():
    from morie.fn.sbpiv import sbpiv
    assert sbpiv is sbpiv
