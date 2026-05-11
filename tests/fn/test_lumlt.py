"""Tests for lumlt."""
import numpy as np
import pytest
from morie.fn.lumlt import lumlt


def test_lumlt_basic():
    result = lumlt()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "LU-MultiField"


def test_lumlt_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = lumlt(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_lumlt_no_data():
    result = lumlt(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_lumlt_alias():
    from morie.fn.lumlt import lumlt
    assert lumlt is lumlt
