"""Tests for cdval."""
import numpy as np
import pytest
from moirais.fn.cdval import cdval


def test_cdval_basic():
    result = cdval()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ConditionalSim-EType"


def test_cdval_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = cdval(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_cdval_no_data():
    result = cdval(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_cdval_alias():
    from moirais.fn.cdval import cdval
    assert cdval is cdval
