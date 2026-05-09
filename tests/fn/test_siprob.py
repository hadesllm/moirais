"""Tests for siprob."""
import numpy as np
import pytest
from moirais.fn.siprob import siprob


def test_siprob_basic():
    result = siprob()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SIS-ProbMap"


def test_siprob_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = siprob(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_siprob_no_data():
    result = siprob(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_siprob_alias():
    from moirais.fn.siprob import siprob
    assert siprob is siprob
