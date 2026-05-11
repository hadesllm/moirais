"""Tests for rng244.rangayyan_ch4_complex_cepstrum_closed_form."""
import numpy as np
import pytest
from morie.fn.rng244 import rangayyan_ch4_complex_cepstrum_closed_form


def test_rng244_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    c_k = np.random.default_rng(42).normal(0, 1, 100)
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    M_I = np.random.default_rng(42).normal(0, 1, 100)
    M_O = np.random.default_rng(42).normal(0, 1, 100)
    N_I = np.random.default_rng(42).normal(0, 1, 100)
    N_O = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_complex_cepstrum_closed_form(A, a_k, b_k, c_k, d_k, M_I, M_O, N_I, N_O, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng244_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    c_k = np.random.default_rng(42).normal(0, 1, 100)
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    M_I = np.random.default_rng(42).normal(0, 1, 100)
    M_O = np.random.default_rng(42).normal(0, 1, 100)
    N_I = np.random.default_rng(42).normal(0, 1, 100)
    N_O = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_complex_cepstrum_closed_form(A, a_k, b_k, c_k, d_k, M_I, M_O, N_I, N_O, n)
    assert isinstance(result, dict)
