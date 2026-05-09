"""Tests for stfta.py - STFT analysis."""
import numpy as np
import pytest
from moirais.fn.stfta import stft_analysis, stfta


def test_stft_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = stft_analysis(x)
    assert result.name == "stft"
    assert "spectrogram" in result.extra
    assert "times" in result.extra
    assert "freqs" in result.extra


def test_stft_spectrogram_shape():
    x = np.random.default_rng(42).standard_normal(512)
    result = stft_analysis(x, window_size=64, hop=32)
    S = result.extra["spectrogram"]
    assert S.ndim == 2


def test_stft_freqs_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = stft_analysis(x)
    freqs = result.extra["freqs"]
    assert len(freqs) > 0
    assert freqs[0] >= 0


def test_stfta_alias():
    x = np.random.default_rng(42).standard_normal(256)
    result = stfta(x)
    assert result.name == "stft"
