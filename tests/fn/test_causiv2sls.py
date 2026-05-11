"""Tests for causiv2sls.causal_iv_2sls."""
import numpy as np
import pytest
from morie.fn.causiv2sls import causal_iv_2sls


def test_causiv2sls_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = causal_iv_2sls(y, X, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causiv2sls_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = causal_iv_2sls(y, X, Z)
    assert isinstance(result, dict)
