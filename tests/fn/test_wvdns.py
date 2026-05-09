"""Tests for wvdns.py - Wavelet denoising."""
import numpy as np
from moirais.fn.wvdns import wavelet_denoise, wvdns


def test_wvdns_returns_descriptive_result():
    rng = np.random.default_rng(42)
    x = np.sin(np.linspace(0, 4 * np.pi, 256)) + 0.5 * rng.standard_normal(256)
    result = wavelet_denoise(x, wavelet="haar")
    assert result.name == "wavelet_denoise"
    assert "denoised" in result.extra


def test_wvdns_reduces_noise():
    rng = np.random.default_rng(42)
    clean = np.sin(np.linspace(0, 4 * np.pi, 256))
    noisy = clean + 2.0 * rng.standard_normal(256)
    result = wavelet_denoise(noisy, wavelet="haar")
    denoised = result.extra["denoised"][:len(clean)]
    assert "threshold" in result.extra
    assert result.extra["threshold"] > 0


def test_wvdns_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = wvdns(x, wavelet="haar")
    assert result.name == "wavelet_denoise"
