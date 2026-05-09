"""Tests for spcsts."""
import numpy as np
import pytest
from moirais.fn.spcsts import spcsts


def test_spcsts_basic():
    result = spcsts()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpectralSim-WoodChan"


def test_spcsts_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = spcsts(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_spcsts_no_data():
    result = spcsts(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_spcsts_alias():
    from moirais.fn.spcsts import spcsts
    assert spcsts is spcsts
