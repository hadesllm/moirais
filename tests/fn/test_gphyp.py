"""Tests for gphyp."""
import numpy as np
import pytest
from moirais.fn.gphyp import gphyp


def test_gphyp_basic():
    result = gphyp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GP-Hyperparameters"


def test_gphyp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = gphyp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_gphyp_no_data():
    result = gphyp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_gphyp_alias():
    from moirais.fn.gphyp import gphyp
    assert gphyp is gphyp
