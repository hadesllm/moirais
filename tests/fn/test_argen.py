"""Tests for argen.py - AR process generation."""
import numpy as np
from morie.fn.argen import ar_generate_fn, argen


def test_argen_returns_result():
    result = ar_generate_fn(np.array([0.5, -0.3]), sigma2=1.0, N=100, seed=42)
    assert result.name == "ar_generate"
    assert len(result.extra["signal"]) == 100


def test_argen_reproducible():
    r1 = ar_generate_fn(np.array([0.5]), N=50, seed=42)
    r2 = ar_generate_fn(np.array([0.5]), N=50, seed=42)
    np.testing.assert_array_equal(r1.extra["signal"], r2.extra["signal"])


def test_argen_alias():
    result = argen(np.array([0.3]), N=50, seed=42)
    assert result.name == "ar_generate"
