"""Tests for stvario."""
import numpy as np
import pytest
from moirais.fn.stvario import stvario


def test_stvario_basic():
    result = stvario()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Variogram"


def test_stvario_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stvario(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stvario_no_data():
    result = stvario(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stvario_alias():
    from moirais.fn.stvario import stvario
    assert stvario is stvario
