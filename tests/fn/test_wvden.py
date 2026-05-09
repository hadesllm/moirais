"""Tests for wvden.py - Wavelet denoising."""
import numpy as np
from moirais.fn.wvden import wavelet_denoise, wvden


def test_denoise_returns_result():
    rng = np.random.default_rng(42)
    x = np.sin(np.linspace(0, 4 * np.pi, 256)) + rng.standard_normal(256) * 0.3
    result = wavelet_denoise(x)
    assert result.name == "wavelet_denoise"
    assert "denoised" in result.extra
    assert len(result.extra["denoised"]) == len(x)


def test_denoise_hard_threshold():
    rng = np.random.default_rng(42)
    x = np.sin(np.linspace(0, 4 * np.pi, 128)) + rng.standard_normal(128) * 0.5
    result = wavelet_denoise(x, threshold="hard")
    assert result.extra["method"] == "hard"


def test_denoise_produces_output():
    rng = np.random.default_rng(42)
    clean = np.sin(np.linspace(0, 4 * np.pi, 256))
    noisy = clean + rng.standard_normal(256) * 0.5
    result = wavelet_denoise(noisy)
    denoised = result.extra["denoised"]
    assert denoised is not None
    assert not np.allclose(denoised, 0)
    assert result.extra["threshold_value"] > 0


def test_denoise_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = wvden(x)
    assert result.name == "wavelet_denoise"
