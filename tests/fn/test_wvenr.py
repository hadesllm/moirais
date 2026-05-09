"""Tests for wvenr.py - Wavelet energy."""
import numpy as np
from moirais.fn.wvenr import wavelet_energy, wvenr


def test_energy_returns_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = wavelet_energy(x, level=3)
    assert result.name == "wavelet_energy"
    assert "energies" in result.extra


def test_energy_band_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = wavelet_energy(x, level=3)
    assert len(result.extra["energies"]) == 4
    assert len(result.extra["labels"]) == 4


def test_energy_ratios_sum():
    x = np.random.default_rng(42).standard_normal(256)
    result = wavelet_energy(x, level=3)
    assert abs(sum(result.extra["ratios"]) - 1.0) < 0.05


def test_energy_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = wvenr(x, level=2)
    assert result.name == "wavelet_energy"
