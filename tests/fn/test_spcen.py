"""Tests for spcen.py - spectral entropy."""
import numpy as np
import pytest
from morie.fn.spcen import spectral_entropy_fn, spcen


def test_spectral_entropy_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_entropy_fn(x)
    assert result.name == "spectral_entropy"
    assert isinstance(result.value, float)


def test_spectral_entropy_nonnegative():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_entropy_fn(x)
    assert result.value >= 0.0


def test_spectral_entropy_white_noise_positive():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1024)
    result = spectral_entropy_fn(x)
    assert result.value > 0.0


def test_spcen_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = spcen(x)
    assert result.name == "spectral_entropy"
