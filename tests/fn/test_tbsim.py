"""Tests for tbsim."""
import numpy as np
import pytest
from moirais.fn.tbsim import tbsim


def test_tbsim_basic():
    result = tbsim()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TurningBands2D"


def test_tbsim_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tbsim(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tbsim_no_data():
    result = tbsim(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tbsim_alias():
    from moirais.fn.tbsim import tbsim
    assert tbsim is tbsim
