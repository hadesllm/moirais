"""Tests for stprod."""
import numpy as np
import pytest
from morie.fn.stprod import stprod


def test_stprod_basic():
    result = stprod()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Product"


def test_stprod_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stprod(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stprod_no_data():
    result = stprod(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stprod_alias():
    from morie.fn.stprod import stprod
    assert stprod is stprod
