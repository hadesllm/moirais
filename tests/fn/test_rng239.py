"""Tests for rng239.rangayyan_ch4_rational_z_transform_form."""
import numpy as np
import pytest
from moirais.fn.rng239 import rangayyan_ch4_rational_z_transform_form


def test_rng239_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    z = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    c_k = np.random.default_rng(42).normal(0, 1, 100)
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    M_I = np.random.default_rng(42).normal(0, 1, 100)
    M_O = np.random.default_rng(42).normal(0, 1, 100)
    N_I = np.random.default_rng(42).normal(0, 1, 100)
    N_O = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_rational_z_transform_form(A, z, r, a_k, b_k, c_k, d_k, M_I, M_O, N_I, N_O)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng239_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    z = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    c_k = np.random.default_rng(42).normal(0, 1, 100)
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    M_I = np.random.default_rng(42).normal(0, 1, 100)
    M_O = np.random.default_rng(42).normal(0, 1, 100)
    N_I = np.random.default_rng(42).normal(0, 1, 100)
    N_O = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_rational_z_transform_form(A, z, r, a_k, b_k, c_k, d_k, M_I, M_O, N_I, N_O)
    assert isinstance(result, dict)
