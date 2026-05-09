"""Tests for emdsg -- Empirical Mode Decomposition."""
import numpy as np
from moirais.fn.emdsg import emd
from moirais.fn._containers import DescriptiveResult


def test_emd_basic():
    rng = np.random.default_rng(42)
    fs = 500
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
    x += rng.standard_normal(len(t)) * 0.05
    result = emd(x, max_imfs=5)
    assert isinstance(result, DescriptiveResult)
    assert "imfs" in result.extra
    assert "residue" in result.extra
    assert "n_imfs" in result.extra
    assert "sift_counts" in result.extra
    assert "is_imf" in result.extra


def test_emd_reconstruction():
    rng = np.random.default_rng(7)
    fs = 200
    t = np.arange(0, 2.0, 1 / fs)
    x = np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 40 * t)
    result = emd(x, max_imfs=10)
    imfs = result.extra["imfs"]
    residue = result.extra["residue"]
    reconstructed = sum(imfs) + residue
    assert np.allclose(x, reconstructed, atol=1e-6)


def test_emd_imf_count():
    rng = np.random.default_rng(99)
    x = rng.standard_normal(256)
    result = emd(x, max_imfs=4)
    assert result.extra["n_imfs"] <= 4
    assert len(result.extra["imfs"]) == result.extra["n_imfs"]
    assert len(result.extra["sift_counts"]) == result.extra["n_imfs"]
    assert len(result.extra["is_imf"]) == result.extra["n_imfs"]


def test_emd_pure_sine():
    fs = 100
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 5 * t)
    result = emd(x, max_imfs=3)
    assert result.extra["n_imfs"] >= 1
