"""Tests for tbens."""
import numpy as np
import pytest
from morie.fn.tbens import tbens


def test_tbens_basic():
    result = tbens()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TurningBands-Ensemble"


def test_tbens_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tbens(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tbens_no_data():
    result = tbens(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tbens_alias():
    from morie.fn.tbens import tbens
    assert tbens is tbens
