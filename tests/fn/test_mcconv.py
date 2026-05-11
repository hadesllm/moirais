"""Tests for mcconv."""
import numpy as np
import pytest
from morie.fn.mcconv import mcconv


def test_mcconv_basic():
    result = mcconv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "MC-Convergence"


def test_mcconv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = mcconv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_mcconv_no_data():
    result = mcconv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_mcconv_alias():
    from morie.fn.mcconv import mcconv
    assert mcconv is mcconv
