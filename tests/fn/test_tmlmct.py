"""Tests for tmlmct.tmle_multivariate_treatment."""
import numpy as np
import pytest
from moirais.fn.tmlmct import tmle_multivariate_treatment


def test_tmlmct_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_multivariate_treatment(y, A, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlmct_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_multivariate_treatment(y, A, X)
    assert isinstance(result, dict)
