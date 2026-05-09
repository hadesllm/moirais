"""Tests for ridgx.py - Ridge extraction."""
import numpy as np
from moirais.fn.ridgx import ridge_extract, ridgx


def test_ridgx_returns_descriptive_result():
    tfr = np.random.default_rng(42).random((32, 64))
    result = ridge_extract(tfr, fs=100.0)
    assert result.name == "ridge_extract"
    assert "ridge_indices" in result.extra


def test_ridgx_ridge_length():
    tfr = np.random.default_rng(42).random((16, 32))
    result = ridge_extract(tfr)
    assert len(result.extra["ridge_indices"]) == 32


def test_ridgx_alias():
    tfr = np.random.default_rng(42).random((8, 16))
    result = ridgx(tfr)
    assert result.name == "ridge_extract"
