"""Tests for sgolay — Savitzky-Golay smoothing."""
import numpy as np
from moirais.fn.sgolay import savgol_smooth
from moirais.fn._containers import SignalResult


def test_sgolay_basic(rng):
    x = np.sin(np.linspace(0, 4 * np.pi, 200)) + rng.standard_normal(200) * 0.3
    result = savgol_smooth(x, window=11, polyorder=3)
    assert isinstance(result, SignalResult)
    assert result.n_samples == len(x)


def test_sgolay_reduces_noise(rng):
    clean = np.sin(np.linspace(0, 4 * np.pi, 200))
    noisy = clean + rng.standard_normal(200) * 0.5
    result = savgol_smooth(noisy, window=21, polyorder=3)
    noise_before = np.std(noisy - clean)
    noise_after = np.std(result.filtered - clean)
    assert noise_after < noise_before
