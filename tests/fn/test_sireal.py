"""Tests for sireal."""
import numpy as np
import pytest
from moirais.fn.sireal import sireal


def test_sireal_basic():
    result = sireal()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SIS-Categorical"


def test_sireal_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sireal(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sireal_no_data():
    result = sireal(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sireal_alias():
    from moirais.fn.sireal import sireal
    assert sireal is sireal
