"""Tests for sivar."""
import numpy as np
import pytest
from moirais.fn.sivar import sivar


def test_sivar_basic():
    result = sivar()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SIS-VarioRepro"


def test_sivar_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sivar(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sivar_no_data():
    result = sivar(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sivar_alias():
    from moirais.fn.sivar import sivar
    assert sivar is sivar
