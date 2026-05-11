"""Tests for idwp."""
import numpy as np
import pytest
from morie.fn.idwp import idwp


def test_idwp_basic():
    result = idwp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "IDW-Power"


def test_idwp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = idwp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_idwp_no_data():
    result = idwp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_idwp_alias():
    from morie.fn.idwp import idwp
    assert idwp is idwp
