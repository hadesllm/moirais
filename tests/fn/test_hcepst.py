"""Tests for hcepst — Complex cepstrum."""
import numpy as np
from morie.fn.hcepst import complex_cepstrum
from morie.fn._containers import SignalResult


def test_hcepst_basic(rng):
    x = rng.standard_normal(256)
    result = complex_cepstrum(x)
    assert isinstance(result, SignalResult)
    assert result.filtered is not None


def test_hcepst_length():
    x = np.random.default_rng(0).standard_normal(100)
    result = complex_cepstrum(x, n_fft=256)
    assert len(result.filtered) == 256
