"""Tests for luprec."""
import numpy as np
import pytest
from moirais.fn.luprec import luprec


def test_luprec_basic():
    result = luprec()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "LU-Preconditioned"


def test_luprec_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = luprec(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_luprec_no_data():
    result = luprec(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_luprec_alias():
    from moirais.fn.luprec import luprec
    assert luprec is luprec
