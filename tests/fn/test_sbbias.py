"""Tests for sbbias."""
import numpy as np
import pytest
from morie.fn.sbbias import sbbias


def test_sbbias_basic():
    result = sbbias()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpatialBootstrap-Bias"


def test_sbbias_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sbbias(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sbbias_no_data():
    result = sbbias(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sbbias_alias():
    from morie.fn.sbbias import sbbias
    assert sbbias is sbbias
