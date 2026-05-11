"""Tests for spcnst."""
import numpy as np
import pytest
from morie.fn.spcnst import spcnst


def test_spcnst_basic():
    result = spcnst()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpectralSim-NonStationary"


def test_spcnst_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = spcnst(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_spcnst_no_data():
    result = spcnst(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_spcnst_alias():
    from morie.fn.spcnst import spcnst
    assert spcnst is spcnst
