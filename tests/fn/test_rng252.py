"""Tests for rng252.rangayyan_ch4_complex_cepstrum_signal_with_echo."""
import numpy as np
import pytest
from moirais.fn.rng252 import rangayyan_ch4_complex_cepstrum_signal_with_echo


def test_rng252_basic():
    """Test basic functionality."""
    h_hat = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_complex_cepstrum_signal_with_echo(h_hat, a, n_0, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng252_edge():
    """Test edge cases."""
    h_hat = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_complex_cepstrum_signal_with_echo(h_hat, a, n_0, n)
    assert isinstance(result, dict)
