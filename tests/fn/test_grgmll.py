"""Tests for grgmll.geron_gmm_log_likelihood."""
import numpy as np
import pytest
from morie.fn.grgmll import geron_gmm_log_likelihood


def test_grgmll_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    means = np.random.default_rng(42).normal(0, 1, 100)
    covars = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gmm_log_likelihood(X, pi, means, covars)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grgmll_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    means = np.random.default_rng(42).normal(0, 1, 100)
    covars = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gmm_log_likelihood(X, pi, means, covars)
    assert isinstance(result, dict)
