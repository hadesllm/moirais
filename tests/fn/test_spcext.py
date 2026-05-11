"""Tests for spcext."""
import numpy as np
import pytest
from morie.fn.spcext import spcext


def test_spcext_basic():
    result = spcext()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpectralSim-ExactCirc"


def test_spcext_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = spcext(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_spcext_no_data():
    result = spcext(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_spcext_alias():
    from morie.fn.spcext import spcext
    assert spcext is spcext
