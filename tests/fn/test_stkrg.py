"""Tests for stkrg."""
import numpy as np
import pytest
from morie.fn.stkrg import stkrg


def test_stkrg_basic():
    result = stkrg()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Kriging-Ordinary"


def test_stkrg_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stkrg(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stkrg_no_data():
    result = stkrg(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stkrg_alias():
    from morie.fn.stkrg import stkrg
    assert stkrg is stkrg
