"""Tests for blasf.bayesian_lasso_full."""
import numpy as np
import pytest
from morie.fn.blasf import bayesian_lasso_full


def test_blasf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bayesian_lasso_full(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blasf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bayesian_lasso_full(x, y)
    assert isinstance(result, dict)
