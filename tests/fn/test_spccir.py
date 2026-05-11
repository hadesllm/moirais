"""Tests for spccir."""
import numpy as np
import pytest
from morie.fn.spccir import spccir


def test_spccir_basic():
    result = spccir()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpectralSim-Circulant"


def test_spccir_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = spccir(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_spccir_no_data():
    result = spccir(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_spccir_alias():
    from morie.fn.spccir import spccir
    assert spccir is spccir
