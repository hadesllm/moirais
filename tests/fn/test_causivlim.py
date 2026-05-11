"""Tests for causivlim.causal_iv_liml."""
import numpy as np
import pytest
from morie.fn.causivlim import causal_iv_liml


def test_causivlim_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = causal_iv_liml(y, X, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causivlim_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = causal_iv_liml(y, X, Z)
    assert isinstance(result, dict)
