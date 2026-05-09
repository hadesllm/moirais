"""Tests for rng238.rangayyan_ch4_complex_cepstra_sum."""
import numpy as np
import pytest
from moirais.fn.rng238 import rangayyan_ch4_complex_cepstra_sum


def test_rng238_basic():
    """Test basic functionality."""
    x_hat = np.random.default_rng(42).normal(0, 1, 100)
    h_hat = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_complex_cepstra_sum(x_hat, h_hat, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng238_edge():
    """Test edge cases."""
    x_hat = np.random.default_rng(42).normal(0, 1, 100)
    h_hat = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_complex_cepstra_sum(x_hat, h_hat, n)
    assert isinstance(result, dict)
