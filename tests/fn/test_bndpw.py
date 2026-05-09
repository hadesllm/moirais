"""Tests for bndpw.py - band power."""
import numpy as np
import pytest
from moirais.fn.bndpw import band_power_fn, bndpw


def test_band_power_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = band_power_fn(x, fs=100.0)
    assert result.name == "band_power"
    assert isinstance(result.value, float)


def test_band_power_nonnegative():
    x = np.random.default_rng(42).standard_normal(256)
    result = band_power_fn(x, fs=100.0, f_low=1.0, f_high=10.0)
    assert result.value >= 0


def test_band_power_wider_band_ge_narrower():
    x = np.random.default_rng(42).standard_normal(256)
    r_wide = band_power_fn(x, fs=100.0, f_low=1.0, f_high=20.0)
    r_narrow = band_power_fn(x, fs=100.0, f_low=5.0, f_high=10.0)
    assert r_wide.value >= r_narrow.value


def test_bndpw_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = bndpw(x, fs=10.0, f_low=0.5, f_high=4.0)
    assert result.name == "band_power"
