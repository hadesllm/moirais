"""Tests for sgnbr."""
import numpy as np
import pytest
from moirais.fn.sgnbr import sgnbr


def test_sgnbr_basic():
    result = sgnbr()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-Neighborhood"


def test_sgnbr_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sgnbr(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sgnbr_no_data():
    result = sgnbr(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sgnbr_alias():
    from moirais.fn.sgnbr import sgnbr
    assert sgnbr is sgnbr
