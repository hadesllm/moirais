"""Tests for chlpow."""
import numpy as np
import pytest
from moirais.fn.chlpow import chlpow


def test_chlpow_basic():
    result = chlpow()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Cholesky-Power"


def test_chlpow_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlpow(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlpow_no_data():
    result = chlpow(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlpow_alias():
    from moirais.fn.chlpow import chlpow
    assert chlpow is chlpow
