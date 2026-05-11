"""Tests for spgrm.py - spectrogram."""
import numpy as np
import pytest
from morie.fn.spgrm import spectrogram_fn, spgrm


def test_spectrogram_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectrogram_fn(x)
    assert result.name == "spectrogram"
    assert "power" in result.extra
    assert "times" in result.extra
    assert "freqs" in result.extra


def test_spectrogram_power_nonnegative():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectrogram_fn(x)
    assert np.all(result.extra["power"] >= 0)


def test_spectrogram_shape_2d():
    x = np.random.default_rng(42).standard_normal(512)
    result = spectrogram_fn(x, window_size=64, hop=32)
    assert result.extra["power"].ndim == 2


def test_spgrm_alias():
    x = np.random.default_rng(42).standard_normal(256)
    result = spgrm(x)
    assert result.name == "spectrogram"
