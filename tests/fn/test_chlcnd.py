"""Tests for chlcnd."""
import numpy as np
import pytest
from moirais.fn.chlcnd import chlcnd


def test_chlcnd_basic():
    result = chlcnd()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "CholeskyCondGrid"


def test_chlcnd_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlcnd(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlcnd_no_data():
    result = chlcnd(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlcnd_alias():
    from moirais.fn.chlcnd import chlcnd
    assert chlcnd is chlcnd
