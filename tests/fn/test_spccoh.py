"""Tests for spccoh."""
import numpy as np
import pytest
from morie.fn.spccoh import spccoh


def test_spccoh_basic():
    result = spccoh()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpectralSim-Coherence"


def test_spccoh_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = spccoh(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_spccoh_no_data():
    result = spccoh(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_spccoh_alias():
    from morie.fn.spccoh import spccoh
    assert spccoh is spccoh
