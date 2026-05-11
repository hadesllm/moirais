"""Tests for sgprob."""
import numpy as np
import pytest
from morie.fn.sgprob import sgprob


def test_sgprob_basic():
    result = sgprob()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-Probability"


def test_sgprob_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sgprob(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sgprob_no_data():
    result = sgprob(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sgprob_alias():
    from morie.fn.sgprob import sgprob
    assert sgprob is sgprob
