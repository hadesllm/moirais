"""Tests for stkrgs."""
import numpy as np
import pytest
from moirais.fn.stkrgs import stkrgs


def test_stkrgs_basic():
    result = stkrgs()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Kriging-Simple"


def test_stkrgs_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stkrgs(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stkrgs_no_data():
    result = stkrgs(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stkrgs_alias():
    from moirais.fn.stkrgs import stkrgs
    assert stkrgs is stkrgs
