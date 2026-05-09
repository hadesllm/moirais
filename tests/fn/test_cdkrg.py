"""Tests for cdkrg."""
import numpy as np
import pytest
from moirais.fn.cdkrg import cdkrg


def test_cdkrg_basic():
    result = cdkrg()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ConditionalSim-Kriging"


def test_cdkrg_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = cdkrg(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_cdkrg_no_data():
    result = cdkrg(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_cdkrg_alias():
    from moirais.fn.cdkrg import cdkrg
    assert cdkrg is cdkrg
