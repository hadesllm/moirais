"""Tests for stqft -- Short-time quadratic frequency transform."""
import numpy as np
from morie.fn.stqft import stqft
from morie.fn._containers import DescriptiveResult


def test_stqft_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(512)
    result = stqft(x, fs=1000.0, nperseg=64)
    assert isinstance(result, DescriptiveResult)
    assert "stqf" in result.extra


def test_stqft_shape():
    x = np.random.default_rng(7).standard_normal(256)
    result = stqft(x, nperseg=64)
    stq = result.extra["stqf"]
    assert stq.shape[0] == 64 // 2 + 1


def test_stqft_nonnegative():
    x = np.random.default_rng(99).standard_normal(128)
    result = stqft(x, nperseg=32)
    assert np.all(result.extra["stqf"] >= 0)
