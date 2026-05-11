"""Tests for cepst — Real cepstrum."""
import numpy as np
from morie.fn.cepst import real_cepstrum
from morie.fn._containers import SignalResult


def test_cepst_basic(rng):
    x = rng.standard_normal(256)
    result = real_cepstrum(x)
    assert isinstance(result, SignalResult)
    assert result.filtered is not None


def test_cepst_echo_detection():
    n = 512
    x = np.zeros(n)
    x[0] = 1.0
    x[50] = 0.5
    result = real_cepstrum(x)
    ceps = result.filtered
    peak_idx = np.argmax(np.abs(ceps[10:100])) + 10
    assert abs(peak_idx - 50) < 5
