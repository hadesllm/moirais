"""Tests for coher -- Coherence between two signals."""
import numpy as np
from moirais.fn.coher import coherence
from moirais.fn._containers import DescriptiveResult


def test_coherence_identical_signals():
    rng = np.random.default_rng(42)
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 50 * t)
    result = coherence(x, x, fs=fs, nperseg=256)
    assert isinstance(result, DescriptiveResult)
    coh = result.extra["coherence"]
    assert np.all(coh >= 0)
    assert np.max(coh) > 0.9


def test_coherence_uncorrelated():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    result = coherence(x, y, fs=1.0, nperseg=256)
    coh = result.extra["coherence"]
    assert np.mean(coh) < 0.5


def test_coherence_shared_component():
    rng = np.random.default_rng(7)
    fs = 500
    t = np.arange(0, 2.0, 1 / fs)
    shared = np.sin(2 * np.pi * 30 * t)
    x = shared + rng.standard_normal(len(t)) * 0.3
    y = shared + rng.standard_normal(len(t)) * 0.3
    result = coherence(x, y, fs=fs, nperseg=256)
    assert "frequencies" in result.extra
    assert "cross_spectrum" in result.extra
    coh_vals = result.extra["coherence"]
    freqs = result.extra["frequencies"]
    near_30 = np.abs(freqs - 30) < 5
    assert np.max(coh_vals[near_30]) > 0.5
