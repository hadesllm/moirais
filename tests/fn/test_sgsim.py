"""Tests for sgsim."""
import numpy as np
import pytest
from morie.fn.sgsim import sgsim


def test_sgsim_basic():
    result = sgsim()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS"


def test_sgsim_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sgsim(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sgsim_no_data():
    result = sgsim(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sgsim_alias():
    from morie.fn.sgsim import sgsim
    assert sgsim is sgsim
