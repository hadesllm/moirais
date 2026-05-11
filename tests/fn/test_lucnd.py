"""Tests for lucnd."""
import numpy as np
import pytest
from morie.fn.lucnd import lucnd


def test_lucnd_basic():
    result = lucnd()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "LU-Conditional"


def test_lucnd_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = lucnd(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_lucnd_no_data():
    result = lucnd(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_lucnd_alias():
    from morie.fn.lucnd import lucnd
    assert lucnd is lucnd
