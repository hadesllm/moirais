"""Tests for chlvar."""
import numpy as np
import pytest
from morie.fn.chlvar import chlvar


def test_chlvar_basic():
    result = chlvar()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "CholeskyVariance"


def test_chlvar_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlvar(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlvar_no_data():
    result = chlvar(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlvar_alias():
    from morie.fn.chlvar import chlvar
    assert chlvar is chlvar
