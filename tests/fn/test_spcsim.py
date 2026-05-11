"""Tests for spcsim."""
import numpy as np
import pytest
from morie.fn.spcsim import spcsim


def test_spcsim_basic():
    result = spcsim()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpectralSim-FFT"


def test_spcsim_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = spcsim(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_spcsim_no_data():
    result = spcsim(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_spcsim_alias():
    from morie.fn.spcsim import spcsim
    assert spcsim is spcsim
