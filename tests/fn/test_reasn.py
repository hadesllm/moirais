"""Tests for reasn.py - Reassigned STFT."""
import numpy as np
from moirais.fn.reasn import reassigned_stft, reasn


def test_reasn_returns_descriptive_result():
    x = np.sin(np.linspace(0, 4 * np.pi, 512))
    result = reassigned_stft(x, fs=512.0, nperseg=64)
    assert result.name == "reassigned_stft"
    assert "spectrogram" in result.extra


def test_reasn_spectrogram_shape():
    x = np.random.default_rng(42).standard_normal(256)
    result = reassigned_stft(x, fs=256.0, nperseg=64)
    spec = result.extra["spectrogram"]
    assert spec.ndim == 2


def test_reasn_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = reasn(x, nperseg=32)
    assert result.name == "reassigned_stft"
