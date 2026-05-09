"""Tests for rfpois."""
import numpy as np
import pytest
from moirais.fn.rfpois import rfpois


def test_rfpois_basic():
    result = rfpois()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-Poisson"


def test_rfpois_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfpois(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfpois_no_data():
    result = rfpois(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfpois_alias():
    from moirais.fn.rfpois import rfpois
    assert rfpois is rfpois
