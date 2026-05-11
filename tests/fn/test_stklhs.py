"""Tests for stklhs."""
import numpy as np
import pytest
from morie.fn.stklhs import stklhs


def test_stklhs_basic():
    result = stklhs()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Kriging-LhOut"


def test_stklhs_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stklhs(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stklhs_no_data():
    result = stklhs(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stklhs_alias():
    from morie.fn.stklhs import stklhs
    assert stklhs is stklhs
