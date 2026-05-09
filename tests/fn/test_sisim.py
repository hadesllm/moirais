"""Tests for sisim."""
import numpy as np
import pytest
from moirais.fn.sisim import sisim


def test_sisim_basic():
    result = sisim()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SIS"


def test_sisim_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sisim(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sisim_no_data():
    result = sisim(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sisim_alias():
    from moirais.fn.sisim import sisim
    assert sisim is sisim
