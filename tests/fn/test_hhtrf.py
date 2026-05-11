"""Tests for hhtrf -- Hilbert-Huang Transform (full spectrum)."""
import numpy as np
from morie.fn.hhtrf import hilbert_huang_spectrum
from morie.fn._containers import DescriptiveResult


def test_hht_basic():
    rng = np.random.default_rng(42)
    fs = 500
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 80 * t)
    x += rng.standard_normal(len(t)) * 0.1
    result = hilbert_huang_spectrum(x, fs=fs, max_imfs=5)
    assert isinstance(result, DescriptiveResult)
    assert "imfs" in result.extra
    assert "residue" in result.extra
    assert "hilbert_spectrum" in result.extra
    assert "marginal_spectrum" in result.extra
    assert "freq_axis" in result.extra


def test_hht_imfs_sum_to_original():
    rng = np.random.default_rng(7)
    fs = 200
    t = np.arange(0, 2.0, 1 / fs)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)
    result = hilbert_huang_spectrum(x, fs=fs, max_imfs=8)
    imfs = result.extra["imfs"]
    residue = result.extra["residue"]
    reconstructed = sum(imfs) + residue
    assert np.allclose(x, reconstructed, atol=1e-6)


def test_hht_spectrum_shape():
    fs = 256
    t = np.arange(0, 0.5, 1 / fs)
    x = np.sin(2 * np.pi * 40 * t)
    n_bins = 128
    result = hilbert_huang_spectrum(x, fs=fs, n_freq_bins=n_bins)
    hs = result.extra["hilbert_spectrum"]
    assert hs.shape == (n_bins, len(x))
    assert len(result.extra["freq_axis"]) == n_bins


def test_hht_marginal_nonnegative():
    rng = np.random.default_rng(99)
    x = rng.standard_normal(200)
    result = hilbert_huang_spectrum(x, fs=100)
    assert np.all(result.extra["marginal_spectrum"] >= 0)
