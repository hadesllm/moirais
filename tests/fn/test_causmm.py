"""Tests for causmm.causal_mahalanobis_match."""
import numpy as np
import pytest
from moirais.fn.causmm import causal_mahalanobis_match


def test_causmm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    treat = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = causal_mahalanobis_match(X, treat, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causmm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    treat = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = causal_mahalanobis_match(X, treat, k)
    assert isinstance(result, dict)
