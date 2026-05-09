"""Tests for cdbias."""
import numpy as np
import pytest
from moirais.fn.cdbias import cdbias


def test_cdbias_basic():
    result = cdbias()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ConditionalSim-BiasCor"


def test_cdbias_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = cdbias(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_cdbias_no_data():
    result = cdbias(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_cdbias_alias():
    from moirais.fn.cdbias import cdbias
    assert cdbias is cdbias
