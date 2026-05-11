"""Tests for ctfill."""
import numpy as np
import pytest
from morie.fn.ctfill import ctfill


def test_ctfill_basic():
    result = ctfill()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Contour-Filled"


def test_ctfill_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = ctfill(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_ctfill_no_data():
    result = ctfill(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_ctfill_alias():
    from morie.fn.ctfill import ctfill
    assert ctfill is ctfill
