"""Tests for spcpow."""
import numpy as np
import pytest
from morie.fn.spcpow import spcpow


def test_spcpow_basic():
    result = spcpow()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpectralSim-PowerLaw"


def test_spcpow_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = spcpow(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_spcpow_no_data():
    result = spcpow(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_spcpow_alias():
    from morie.fn.spcpow import spcpow
    assert spcpow is spcpow
