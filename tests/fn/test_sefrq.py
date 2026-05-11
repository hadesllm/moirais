"""Tests for sefrq.py - spectral edge frequency."""
import numpy as np
import pytest
from morie.fn.sefrq import spectral_edge_freq, sefrq


def test_spectral_edge_freq_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_edge_freq(x)
    assert result.name == "spectral_edge_frequency"
    assert isinstance(result.value, float)


def test_spectral_edge_freq_95_le_nyquist():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_edge_freq(x, fs=100.0, pct=0.95)
    assert result.value <= 50.0


def test_spectral_edge_freq_50_lt_95():
    x = np.random.default_rng(42).standard_normal(256)
    r50 = spectral_edge_freq(x, fs=100.0, pct=0.5)
    r95 = spectral_edge_freq(x, fs=100.0, pct=0.95)
    assert r50.value <= r95.value


def test_sefrq_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = sefrq(x)
    assert result.name == "spectral_edge_frequency"
