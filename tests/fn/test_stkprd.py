"""Tests for stkprd."""
import numpy as np
import pytest
from morie.fn.stkprd import stkprd


def test_stkprd_basic():
    result = stkprd()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Kriging-TempPred"


def test_stkprd_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stkprd(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stkprd_no_data():
    result = stkprd(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stkprd_alias():
    from morie.fn.stkprd import stkprd
    assert stkprd is stkprd
