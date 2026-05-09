"""Tests for chlgss."""
import numpy as np
import pytest
from moirais.fn.chlgss import chlgss


def test_chlgss_basic():
    result = chlgss()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Cholesky-Gaussian"


def test_chlgss_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlgss(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlgss_no_data():
    result = chlgss(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlgss_alias():
    from moirais.fn.chlgss import chlgss
    assert chlgss is chlgss
