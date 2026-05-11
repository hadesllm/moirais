"""Tests for ctvert."""
import numpy as np
import pytest
from morie.fn.ctvert import ctvert


def test_ctvert_basic():
    result = ctvert()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Contour-Vertex"


def test_ctvert_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = ctvert(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_ctvert_no_data():
    result = ctvert(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_ctvert_alias():
    from morie.fn.ctvert import ctvert
    assert ctvert is ctvert
