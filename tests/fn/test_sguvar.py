"""Tests for sguvar."""
import numpy as np
import pytest
from moirais.fn.sguvar import sguvar


def test_sguvar_basic():
    result = sguvar()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-EType"


def test_sguvar_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sguvar(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sguvar_no_data():
    result = sguvar(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sguvar_alias():
    from moirais.fn.sguvar import sguvar
    assert sguvar is sguvar
