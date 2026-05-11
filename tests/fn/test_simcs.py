"""Tests for simcs."""
import numpy as np
import pytest
from morie.fn.simcs import simcs


def test_simcs_basic():
    result = simcs()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SIS-MarkovChain"


def test_simcs_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = simcs(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_simcs_no_data():
    result = simcs(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_simcs_alias():
    from morie.fn.simcs import simcs
    assert simcs is simcs
