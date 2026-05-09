"""Tests for gpnug."""
import numpy as np
import pytest
from moirais.fn.gpnug import gpnug


def test_gpnug_basic():
    result = gpnug()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GP-Nugget"


def test_gpnug_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = gpnug(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_gpnug_no_data():
    result = gpnug(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_gpnug_alias():
    from moirais.fn.gpnug import gpnug
    assert gpnug is gpnug
