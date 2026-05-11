"""Tests for cttopo."""
import numpy as np
import pytest
from morie.fn.cttopo import cttopo


def test_cttopo_basic():
    result = cttopo()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Contour-Topographic"


def test_cttopo_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = cttopo(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_cttopo_no_data():
    result = cttopo(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_cttopo_alias():
    from morie.fn.cttopo import cttopo
    assert cttopo is cttopo
