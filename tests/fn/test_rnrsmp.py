"""Tests for rnrsmp."""
import numpy as np
import pytest
from moirais.fn.rnrsmp import rnrsmp


def test_rnrsmp_basic():
    result = rnrsmp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-Resample"


def test_rnrsmp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rnrsmp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rnrsmp_no_data():
    result = rnrsmp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rnrsmp_alias():
    from moirais.fn.rnrsmp import rnrsmp
    assert rnrsmp is rnrsmp
