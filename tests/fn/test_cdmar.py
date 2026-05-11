"""Tests for cdmar."""
import numpy as np
import pytest
from morie.fn.cdmar import cdmar


def test_cdmar_basic():
    result = cdmar()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "CoSim-MarkovBayes"


def test_cdmar_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = cdmar(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_cdmar_no_data():
    result = cdmar(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_cdmar_alias():
    from morie.fn.cdmar import cdmar
    assert cdmar is cdmar
