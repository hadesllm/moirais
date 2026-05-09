"""Tests for music.py - MUSIC spectral estimation."""
import numpy as np
from moirais.fn.music import music_spectrum_fn, music


def test_music_returns_result():
    rng = np.random.default_rng(42)
    t = np.arange(256) / 100.0
    x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t) + 0.1 * rng.standard_normal(256)
    result = music_spectrum_fn(x, nsources=2, order=8, nfft=64, fs=100.0)
    assert result.name == "music_spectrum"
    assert len(result.extra["psd"]) == 64


def test_music_psd_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = music_spectrum_fn(x, nsources=1, order=8, nfft=32)
    assert np.all(result.extra["psd"] > 0)


def test_music_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = music(x, nsources=1, order=4, nfft=32)
    assert result.name == "music_spectrum"
