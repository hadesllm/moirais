"""Tests for chlkrg."""
import numpy as np
import pytest
from moirais.fn.chlkrg import chlkrg


def test_chlkrg_basic():
    result = chlkrg()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Cholesky-Kriging"


def test_chlkrg_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlkrg(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlkrg_no_data():
    result = chlkrg(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlkrg_alias():
    from moirais.fn.chlkrg import chlkrg
    assert chlkrg is chlkrg
