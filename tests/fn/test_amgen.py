"""Tests for amgen.py - ARMA process generation."""
import numpy as np
from moirais.fn.amgen import arma_generate_fn, amgen


def test_amgen_returns_result():
    result = arma_generate_fn(
        np.array([0.5]), np.array([0.3]), sigma2=1.0, N=100, seed=42
    )
    assert result.name == "arma_generate"
    assert len(result.extra["signal"]) == 100


def test_amgen_reproducible():
    r1 = arma_generate_fn(np.array([0.5]), np.array([0.3]), N=50, seed=42)
    r2 = arma_generate_fn(np.array([0.5]), np.array([0.3]), N=50, seed=42)
    np.testing.assert_array_equal(r1.extra["signal"], r2.extra["signal"])


def test_amgen_alias():
    result = amgen(np.array([0.3]), np.array([0.2]), N=50, seed=42)
    assert result.name == "arma_generate"
