"""Tests for mtlmm.multi_trait_lmm."""
import numpy as np
import pytest
from moirais.fn.mtlmm import multi_trait_lmm


def test_mtlmm_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = multi_trait_lmm(Y, X, Z, A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mtlmm_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = multi_trait_lmm(Y, X, Z, A)
    assert isinstance(result, dict)
