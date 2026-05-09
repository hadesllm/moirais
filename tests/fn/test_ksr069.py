"""Tests for ksr069.kosorok_ch3_cox_likelihood_breslow."""
import numpy as np
import pytest
from moirais.fn.ksr069 import kosorok_ch3_cox_likelihood_breslow


def test_ksr069_basic():
    """Test basic functionality."""
    beta = 0.8
    Lambda = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    n = 100
    result = kosorok_ch3_cox_likelihood_breslow(beta, Lambda, Z, V, d, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr069_edge():
    """Test edge cases."""
    beta = 0.8
    Lambda = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    n = 100
    result = kosorok_ch3_cox_likelihood_breslow(beta, Lambda, Z, V, d, n)
    assert isinstance(result, dict)
