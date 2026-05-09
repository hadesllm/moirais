"""Tests for lupiv."""
import numpy as np
import pytest
from moirais.fn.lupiv import lupiv


def test_lupiv_basic():
    result = lupiv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "LU-Pivoting"


def test_lupiv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = lupiv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_lupiv_no_data():
    result = lupiv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_lupiv_alias():
    from moirais.fn.lupiv import lupiv
    assert lupiv is lupiv
