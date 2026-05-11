"""Tests for spcani."""
import numpy as np
import pytest
from morie.fn.spcani import spcani


def test_spcani_basic():
    result = spcani()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpectralSim-Anisotropic"


def test_spcani_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = spcani(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_spcani_no_data():
    result = spcani(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_spcani_alias():
    from morie.fn.spcani import spcani
    assert spcani is spcani
