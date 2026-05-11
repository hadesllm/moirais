"""Tests for sghist."""
import numpy as np
import pytest
from morie.fn.sghist import sghist


def test_sghist_basic():
    result = sghist()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-HistRepro"


def test_sghist_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sghist(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sghist_no_data():
    result = sghist(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sghist_alias():
    from morie.fn.sghist import sghist
    assert sghist is sghist
