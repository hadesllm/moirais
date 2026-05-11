"""Tests for morie.fn.mofnb — m-out-of-n bootstrap."""

import numpy as np
import pytest

from morie.fn.mofnb import mofnb


def test_default_m():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    result = mofnb(x, n_boot=200, seed=7)
    assert result["m"] == int(1000 ** (2.0 / 3.0))


def test_custom_m():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = mofnb(x, m=20, n_boot=100, seed=1)
    assert result["m"] == 20


def test_m_too_large_raises():
    with pytest.raises(ValueError, match="m must be <= n"):
        mofnb(np.array([1.0, 2.0]), m=10)


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        mofnb(np.array([]))
