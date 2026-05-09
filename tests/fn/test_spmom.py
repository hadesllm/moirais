"""Tests for spmom.py - spectral moment."""
import numpy as np
import pytest
from moirais.fn.spmom import spectral_moment_fn, spmom


def test_spectral_moment_order0_returns_total_power():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_moment_fn(x, order=0)
    assert result.name == "spectral_moment"
    assert isinstance(result.value, float)
    assert result.value >= 0


def test_spectral_moment_order1_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_moment_fn(x, fs=100.0, order=1)
    assert result.value >= 0


def test_spectral_moment_order2():
    x = np.random.default_rng(42).standard_normal(256)
    result = spectral_moment_fn(x, order=2)
    assert isinstance(result.value, float)


def test_spmom_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = spmom(x)
    assert result.name == "spectral_moment"
