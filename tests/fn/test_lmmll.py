"""Tests for lmmll.lmm_log_likelihood."""
import numpy as np
import pytest
from morie.fn.lmmll import lmm_log_likelihood


def test_lmmll_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = lmm_log_likelihood(y, X, beta, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lmmll_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = lmm_log_likelihood(y, X, beta, V)
    assert isinstance(result, dict)
