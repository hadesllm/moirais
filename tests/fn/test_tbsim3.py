"""Tests for tbsim3."""
import numpy as np
import pytest
from moirais.fn.tbsim3 import tbsim3


def test_tbsim3_basic():
    result = tbsim3()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TurningBands3D"


def test_tbsim3_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tbsim3(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tbsim3_no_data():
    result = tbsim3(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tbsim3_alias():
    from moirais.fn.tbsim3 import tbsim3
    assert tbsim3 is tbsim3
