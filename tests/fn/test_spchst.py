"""Tests for spchst."""
import numpy as np
import pytest
from moirais.fn.spchst import spchst


def test_spchst_basic():
    result = spchst()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpectralSim-HistTrans"


def test_spchst_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = spchst(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_spchst_no_data():
    result = spchst(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_spchst_alias():
    from moirais.fn.spchst import spchst
    assert spchst is spchst
