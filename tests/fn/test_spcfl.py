"""Tests for spcfl.py - spectral flatness."""
import numpy as np
import pytest
from morie.fn.spcfl import spectral_flatness_fn, spcfl


def test_spectral_flatness_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_flatness_fn(x)
    assert result.name == "spectral_flatness"
    assert isinstance(result.value, float)


def test_spectral_flatness_white_noise_near_1():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1024)
    result = spectral_flatness_fn(x)
    assert 0.0 <= result.value <= 1.0


def test_spectral_flatness_tonal_lower():
    t = np.linspace(0, 1, 512)
    x_tone = np.sin(2 * np.pi * 10 * t)
    x_noise = np.random.default_rng(42).standard_normal(512)
    r_tone = spectral_flatness_fn(x_tone)
    r_noise = spectral_flatness_fn(x_noise)
    assert r_tone.value < r_noise.value


def test_spcfl_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = spcfl(x)
    assert result.name == "spectral_flatness"
