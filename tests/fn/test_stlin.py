"""Tests for stlin."""
import numpy as np
import pytest
from moirais.fn.stlin import stlin


def test_stlin_basic():
    result = stlin()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Linear"


def test_stlin_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stlin(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stlin_no_data():
    result = stlin(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stlin_alias():
    from moirais.fn.stlin import stlin
    assert stlin is stlin
