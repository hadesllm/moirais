"""Tests for magen.py - MA process generation."""
import numpy as np
from moirais.fn.magen import ma_generate_fn, magen


def test_magen_returns_result():
    result = ma_generate_fn(np.array([0.5, -0.3]), sigma2=1.0, N=100, seed=42)
    assert result.name == "ma_generate"
    assert len(result.extra["signal"]) == 100


def test_magen_reproducible():
    r1 = ma_generate_fn(np.array([0.5]), N=50, seed=42)
    r2 = ma_generate_fn(np.array([0.5]), N=50, seed=42)
    np.testing.assert_array_equal(r1.extra["signal"], r2.extra["signal"])


def test_magen_alias():
    result = magen(np.array([0.3]), N=50, seed=42)
    assert result.name == "ma_generate"
