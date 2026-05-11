"""Tests for sprto.py - spectral power ratio."""
import numpy as np
import pytest
from morie.fn.sprto import spectral_ratio, sprto


def test_spectral_ratio_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_ratio(x, fs=100.0)
    assert result.name == "spectral_power_ratio"
    assert isinstance(result.value, float)


def test_spectral_ratio_nonnegative():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_ratio(x, fs=100.0, band1=(1.0, 5.0), band2=(5.0, 10.0))
    assert result.value >= 0


def test_spectral_ratio_custom_bands():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_ratio(x, fs=200.0, band1=(0.5, 4.0), band2=(8.0, 12.0))
    assert isinstance(result.value, float)


def test_sprto_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = sprto(x, fs=10.0)
    assert result.name == "spectral_power_ratio"
