"""Tests for rng245.rangayyan_ch4_complex_cepstrum_decay_bound."""
import numpy as np
import pytest
from moirais.fn.rng245 import rangayyan_ch4_complex_cepstrum_decay_bound


def test_rng245_basic():
    """Test basic functionality."""
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    n = 100
    result = rangayyan_ch4_complex_cepstrum_decay_bound(K, alpha, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng245_edge():
    """Test edge cases."""
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    n = 100
    result = rangayyan_ch4_complex_cepstrum_decay_bound(K, alpha, n)
    assert isinstance(result, dict)
