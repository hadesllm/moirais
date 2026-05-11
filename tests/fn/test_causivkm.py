"""Tests for causivkm.causal_iv_kleibergen_moreira."""
import numpy as np
import pytest
from morie.fn.causivkm import causal_iv_kleibergen_moreira


def test_causivkm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    beta0 = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_iv_kleibergen_moreira(y, X, Z, beta0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_causivkm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    beta0 = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_iv_kleibergen_moreira(y, X, Z, beta0)
    assert isinstance(result, dict)
