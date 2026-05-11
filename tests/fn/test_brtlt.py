"""Tests for brtlt.py - Bartlett PSD estimate."""
import numpy as np
import pytest
from morie.fn.brtlt import bartlett_psd_fn, brtlt


def test_bartlett_psd_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = bartlett_psd_fn(x)
    assert result.name == "bartlett_psd"
    assert "freqs" in result.extra
    assert "psd" in result.extra


def test_bartlett_psd_nonnegative():
    x = np.random.default_rng(42).standard_normal(256)
    result = bartlett_psd_fn(x)
    assert np.all(result.extra["psd"] >= 0)


def test_bartlett_psd_with_fs():
    x = np.random.default_rng(42).standard_normal(256)
    result = bartlett_psd_fn(x, fs=100.0, n_segments=4)
    assert len(result.extra["freqs"]) > 0


def test_brtlt_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = brtlt(x)
    assert result.name == "bartlett_psd"
