"""Tests for gpmat."""
import numpy as np
import pytest
from moirais.fn.gpmat import gpmat


def test_gpmat_basic():
    result = gpmat()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GP-Matern52"


def test_gpmat_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = gpmat(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_gpmat_no_data():
    result = gpmat(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_gpmat_alias():
    from moirais.fn.gpmat import gpmat
    assert gpmat is gpmat
