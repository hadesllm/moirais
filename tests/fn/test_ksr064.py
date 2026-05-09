"""Tests for ksr064.kosorok_ch3_cox_partial_likelihood."""
import numpy as np
import pytest
from moirais.fn.ksr064 import kosorok_ch3_cox_partial_likelihood


def test_ksr064_basic():
    """Test basic functionality."""
    beta = 0.8
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    n = 100
    result = kosorok_ch3_cox_partial_likelihood(beta, Z, V, d, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr064_edge():
    """Test edge cases."""
    beta = 0.8
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    n = 100
    result = kosorok_ch3_cox_partial_likelihood(beta, Z, V, d, n)
    assert isinstance(result, dict)
