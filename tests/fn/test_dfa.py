"""Tests for dfa — Detrended fluctuation analysis."""
import numpy as np
from moirais.fn.dfa import detrended_fluctuation
from moirais.fn._containers import DescriptiveResult


def test_dfa_basic(rng):
    x = rng.standard_normal(500)
    result = detrended_fluctuation(x)
    assert isinstance(result, DescriptiveResult)
    assert result.value is not None


def test_dfa_white_noise(rng):
    x = rng.standard_normal(2000)
    result = detrended_fluctuation(x)
    assert 0.3 < result.value < 0.7


def test_dfa_brownian_motion(rng):
    x = np.cumsum(rng.standard_normal(2000))
    result = detrended_fluctuation(x)
    assert 1.2 < result.value < 1.8
