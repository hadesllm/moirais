"""Tests for semfn.py - spectral error measure."""
import numpy as np
import pytest
from morie.fn.semfn import spectral_error_fn, semfn


def test_spectral_error_returns_descriptive_result():
    rng = np.random.default_rng(42)
    x1 = rng.standard_normal(256)
    x2 = rng.standard_normal(256)
    result = spectral_error_fn(x1, x2)
    assert result.name == "spectral_error_measure"
    assert isinstance(result.value, float)


def test_spectral_error_identical_signals_zero():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_error_fn(x, x)
    assert result.value == pytest.approx(0.0, abs=1e-10)


def test_spectral_error_nonnegative():
    rng = np.random.default_rng(42)
    x1 = rng.standard_normal(256)
    x2 = rng.standard_normal(256)
    result = spectral_error_fn(x1, x2)
    assert result.value >= 0


def test_semfn_alias():
    rng = np.random.default_rng(42)
    x1 = rng.standard_normal(64)
    x2 = rng.standard_normal(64)
    result = semfn(x1, x2)
    assert result.name == "spectral_error_measure"
