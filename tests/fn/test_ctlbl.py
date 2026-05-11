"""Tests for ctlbl."""
import numpy as np
import pytest
from morie.fn.ctlbl import ctlbl


def test_ctlbl_basic():
    result = ctlbl()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Contour-Labels"


def test_ctlbl_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = ctlbl(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_ctlbl_no_data():
    result = ctlbl(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_ctlbl_alias():
    from morie.fn.ctlbl import ctlbl
    assert ctlbl is ctlbl
