"""Tests for tbsmp."""
import numpy as np
import pytest
from moirais.fn.tbsmp import tbsmp


def test_tbsmp_basic():
    result = tbsmp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TurningBands-Diagnostic"


def test_tbsmp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tbsmp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tbsmp_no_data():
    result = tbsmp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tbsmp_alias():
    from moirais.fn.tbsmp import tbsmp
    assert tbsmp is tbsmp
