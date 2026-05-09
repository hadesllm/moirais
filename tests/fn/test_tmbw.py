"""Tests for tmbw."""
import numpy as np
import pytest
from moirais.fn.tmbw import tmbw


def test_tmbw_basic():
    result = tmbw()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TempTrend-Butterworth"


def test_tmbw_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tmbw(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tmbw_no_data():
    result = tmbw(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tmbw_alias():
    from moirais.fn.tmbw import tmbw
    assert tmbw is tmbw
