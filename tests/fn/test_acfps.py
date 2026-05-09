"""Tests for acfps.py - ACF from PSD."""
import numpy as np
import pytest
from moirais.fn.acfps import acf_from_psd_fn, acfps


def test_acf_from_psd_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = acf_from_psd_fn(x)
    assert result.name == "acf_from_psd"
    assert "acf" in result.extra


def test_acf_from_psd_is_array():
    x = np.random.default_rng(42).standard_normal(256)
    result = acf_from_psd_fn(x)
    assert isinstance(result.extra["acf"], np.ndarray)
    assert len(result.extra["acf"]) > 0


def test_acf_from_psd_finite():
    x = np.random.default_rng(42).standard_normal(256)
    result = acf_from_psd_fn(x)
    assert np.all(np.isfinite(result.extra["acf"]))


def test_acfps_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = acfps(x)
    assert result.name == "acf_from_psd"
