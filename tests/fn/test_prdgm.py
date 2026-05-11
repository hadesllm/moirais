"""Tests for prdgm.py - periodogram PSD estimate."""
import numpy as np
import pytest
from morie.fn.prdgm import periodogram_estimate, prdgm


def test_periodogram_estimate_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = periodogram_estimate(x)
    assert result.name == "periodogram"
    assert "freqs" in result.extra
    assert "psd" in result.extra


def test_periodogram_estimate_psd_shape():
    x = np.random.default_rng(42).standard_normal(256)
    result = periodogram_estimate(x, fs=100.0)
    assert len(result.extra["freqs"]) == len(result.extra["psd"])


def test_periodogram_estimate_psd_nonnegative():
    x = np.random.default_rng(42).standard_normal(256)
    result = periodogram_estimate(x)
    assert np.all(result.extra["psd"] >= 0)


def test_prdgm_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = prdgm(x)
    assert result.name == "periodogram"
