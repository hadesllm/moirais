"""Tests for mczone."""
import numpy as np
import pytest
from moirais.fn.mczone import mczone


def test_mczone_basic():
    result = mczone()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "MC-Zonal"


def test_mczone_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = mczone(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_mczone_no_data():
    result = mczone(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_mczone_alias():
    from moirais.fn.mczone import mczone
    assert mczone is mczone
