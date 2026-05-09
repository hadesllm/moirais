"""Tests for causivar.causal_iv_anderson_rubin."""
import numpy as np
import pytest
from moirais.fn.causivar import causal_iv_anderson_rubin


def test_causivar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    beta0 = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_iv_anderson_rubin(y, X, Z, beta0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_causivar_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    beta0 = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_iv_anderson_rubin(y, X, Z, beta0)
    assert isinstance(result, dict)
