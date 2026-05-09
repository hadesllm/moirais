"""Tests for pcgenv — PCG Shannon-energy envelope."""
import numpy as np
from moirais.fn.pcgenv import pcg_envelope
from moirais.fn._containers import SignalResult


def test_pcgenv_basic(rng):
    fs = 2000
    t = np.arange(0, 1.0, 1 / fs)
    pcg = np.sin(2 * np.pi * 100 * t) + rng.standard_normal(len(t)) * 0.1
    result = pcg_envelope(pcg, fs)
    assert isinstance(result, SignalResult)
    assert result.n_samples == len(t)


def test_pcgenv_nonnegative(rng):
    fs = 2000
    t = np.arange(0, 1.0, 1 / fs)
    pcg = np.sin(2 * np.pi * 100 * t)
    result = pcg_envelope(pcg, fs)
    assert np.all(result.filtered >= -1e-10)
