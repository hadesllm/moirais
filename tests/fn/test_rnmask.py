"""Tests for rnmask."""
import numpy as np
import pytest
from moirais.fn.rnmask import rnmask


def test_rnmask_basic():
    result = rnmask()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-Mask"


def test_rnmask_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rnmask(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rnmask_no_data():
    result = rnmask(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rnmask_alias():
    from moirais.fn.rnmask import rnmask
    assert rnmask is rnmask
