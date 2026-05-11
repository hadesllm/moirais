"""Tests for tbbnd."""
import numpy as np
import pytest
from morie.fn.tbbnd import tbbnd


def test_tbbnd_basic():
    result = tbbnd()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TurningBands-Lines"


def test_tbbnd_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tbbnd(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tbbnd_no_data():
    result = tbbnd(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tbbnd_alias():
    from morie.fn.tbbnd import tbbnd
    assert tbbnd is tbbnd
