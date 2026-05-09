"""Tests for gppred."""
import numpy as np
import pytest
from moirais.fn.gppred import gppred


def test_gppred_basic():
    result = gppred()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GP-Prediction"


def test_gppred_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = gppred(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_gppred_no_data():
    result = gppred(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_gppred_alias():
    from moirais.fn.gppred import gppred
    assert gppred is gppred
