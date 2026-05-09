"""Tests for nnsib."""
import numpy as np
import pytest
from moirais.fn.nnsib import nnsib


def test_nnsib_basic():
    result = nnsib()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "NatNeighbor-Sibson"


def test_nnsib_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = nnsib(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_nnsib_no_data():
    result = nnsib(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_nnsib_alias():
    from moirais.fn.nnsib import nnsib
    assert nnsib is nnsib
