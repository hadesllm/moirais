"""Tests for mcsrs."""
import numpy as np
import pytest
from moirais.fn.mcsrs import mcsrs


def test_mcsrs_basic():
    result = mcsrs()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "MC-SRS"


def test_mcsrs_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = mcsrs(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_mcsrs_no_data():
    result = mcsrs(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_mcsrs_alias():
    from moirais.fn.mcsrs import mcsrs
    assert mcsrs is mcsrs
