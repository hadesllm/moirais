"""Tests for rnfocl."""
import numpy as np
import pytest
from moirais.fn.rnfocl import rnfocl


def test_rnfocl_basic():
    result = rnfocl()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-Focal"


def test_rnfocl_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rnfocl(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rnfocl_no_data():
    result = rnfocl(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rnfocl_alias():
    from moirais.fn.rnfocl import rnfocl
    assert rnfocl is rnfocl
