"""Tests for stfit."""
import numpy as np
import pytest
from morie.fn.stfit import stfit


def test_stfit_basic():
    result = stfit()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Fitting"


def test_stfit_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stfit(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stfit_no_data():
    result = stfit(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stfit_alias():
    from morie.fn.stfit import stfit
    assert stfit is stfit
