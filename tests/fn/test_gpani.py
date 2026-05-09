"""Tests for gpani."""
import numpy as np
import pytest
from moirais.fn.gpani import gpani


def test_gpani_basic():
    result = gpani()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GP-ARD"


def test_gpani_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = gpani(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_gpani_no_data():
    result = gpani(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_gpani_alias():
    from moirais.fn.gpani import gpani
    assert gpani is gpani
