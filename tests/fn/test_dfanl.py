"""Tests for dfanl -- Detrended fluctuation analysis."""
import numpy as np
from morie.fn.dfanl import dfanl
from morie.fn._containers import DescriptiveResult


def test_dfanl_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    result = dfanl(x)
    assert isinstance(result, DescriptiveResult)
    assert "alpha" in result.extra


def test_dfanl_white_noise():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(2000)
    result = dfanl(x)
    assert 0.2 < result.value < 0.8


def test_dfanl_brownian():
    rng = np.random.default_rng(99)
    x = np.cumsum(rng.standard_normal(2000))
    result = dfanl(x)
    assert result.value > 1.0
