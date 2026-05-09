"""Tests for ceemf.py - Complete EEMD."""
import numpy as np
from moirais.fn.ceemf import ceemd_decompose, ceemf


def test_ceemd_returns_result():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 200)
    x = np.sin(2 * np.pi * 5 * t) + rng.standard_normal(200) * 0.1
    result = ceemd_decompose(x, n_imfs=2, n_trials=5)
    assert result.name == "ceemd_decompose"
    assert "imfs" in result.extra


def test_ceemd_n_trials():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = ceemd_decompose(x, n_imfs=2, n_trials=3)
    assert result.extra["n_trials"] == 3


def test_ceemd_alias():
    x = np.random.default_rng(42).standard_normal(100)
    result = ceemf(x, n_imfs=2, n_trials=3)
    assert result.name == "ceemd_decompose"
