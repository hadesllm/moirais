"""Tests for stckrg."""
import numpy as np
import pytest
from morie.fn.stckrg import stckrg


def test_stckrg_basic():
    result = stckrg()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-CoKriging"


def test_stckrg_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stckrg(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stckrg_no_data():
    result = stckrg(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stckrg_alias():
    from morie.fn.stckrg import stckrg
    assert stckrg is stckrg
