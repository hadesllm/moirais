"""Tests for spcmat."""
import numpy as np
import pytest
from morie.fn.spcmat import spcmat


def test_spcmat_basic():
    result = spcmat()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpectralSim-Matern"


def test_spcmat_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = spcmat(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_spcmat_no_data():
    result = spcmat(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_spcmat_alias():
    from morie.fn.spcmat import spcmat
    assert spcmat is spcmat
