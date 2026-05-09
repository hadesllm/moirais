"""Tests for hdecon — Homomorphic deconvolution."""
import numpy as np
from moirais.fn.hdecon import homomorphic_deconvolve
from moirais.fn._containers import SignalResult


def test_hdecon_basic(rng):
    x = rng.standard_normal(256)
    result = homomorphic_deconvolve(x, cutoff=30)
    assert isinstance(result, SignalResult)
    assert result.filtered is not None
    assert "excitation" in result.extra


def test_hdecon_output_length():
    x = np.random.default_rng(0).standard_normal(100)
    result = homomorphic_deconvolve(x, cutoff=20)
    assert len(result.filtered) == 100
    assert len(result.extra["excitation"]) == 100
