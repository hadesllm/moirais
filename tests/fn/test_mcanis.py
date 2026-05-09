"""Tests for mcanis."""
import numpy as np
import pytest
from moirais.fn.mcanis import mcanis


def test_mcanis_basic():
    result = mcanis()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "MC-Anisotropic"


def test_mcanis_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = mcanis(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_mcanis_no_data():
    result = mcanis(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_mcanis_alias():
    from moirais.fn.mcanis import mcanis
    assert mcanis is mcanis
