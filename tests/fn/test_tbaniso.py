"""Tests for tbaniso."""
import numpy as np
import pytest
from moirais.fn.tbaniso import tbaniso


def test_tbaniso_basic():
    result = tbaniso()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TurningBands-Anisotropy"


def test_tbaniso_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tbaniso(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tbaniso_no_data():
    result = tbaniso(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tbaniso_alias():
    from moirais.fn.tbaniso import tbaniso
    assert tbaniso is tbaniso
