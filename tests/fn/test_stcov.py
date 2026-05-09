"""Tests for stcov."""
import numpy as np
import pytest
from moirais.fn.stcov import stcov


def test_stcov_basic():
    result = stcov()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Covariance-Separable"


def test_stcov_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stcov(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stcov_no_data():
    result = stcov(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stcov_alias():
    from moirais.fn.stcov import stcov
    assert stcov is stcov
