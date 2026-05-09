"""Tests for rflgn."""
import numpy as np
import pytest
from moirais.fn.rflgn import rflgn


def test_rflgn_basic():
    result = rflgn()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-LogNormal"


def test_rflgn_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rflgn(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rflgn_no_data():
    result = rflgn(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rflgn_alias():
    from moirais.fn.rflgn import rflgn
    assert rflgn is rflgn
