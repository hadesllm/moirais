"""Tests for sbbnd."""
import numpy as np
import pytest
from morie.fn.sbbnd import sbbnd


def test_sbbnd_basic():
    result = sbbnd()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpatialBootstrap-Bands"


def test_sbbnd_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sbbnd(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sbbnd_no_data():
    result = sbbnd(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sbbnd_alias():
    from morie.fn.sbbnd import sbbnd
    assert sbbnd is sbbnd
