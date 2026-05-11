"""Tests for nmffn.py - Non-negative Matrix Factorization."""
import numpy as np
import pytest
from morie.fn.nmffn import nmf_fn, nmffn


def test_nmf_returns_descriptive_result():
    V = np.abs(np.random.default_rng(42).standard_normal((10, 20)))
    result = nmf_fn(V)
    assert result.name == "nmf"
    assert "W" in result.extra
    assert "H" in result.extra


def test_nmf_output_shapes():
    V = np.abs(np.random.default_rng(42).standard_normal((10, 20)))
    result = nmf_fn(V, n_components=3)
    W = result.extra["W"]
    H = result.extra["H"]
    assert W.shape[0] == 10
    assert W.shape[1] == 3
    assert H.shape[0] == 3
    assert H.shape[1] == 20


def test_nmf_factors_nonnegative():
    V = np.abs(np.random.default_rng(42).standard_normal((10, 20)))
    result = nmf_fn(V, n_components=3)
    assert np.all(result.extra["W"] >= 0)
    assert np.all(result.extra["H"] >= 0)


def test_nmffn_alias():
    V = np.abs(np.random.default_rng(42).standard_normal((8, 16)))
    result = nmffn(V, n_components=2)
    assert result.name == "nmf"
