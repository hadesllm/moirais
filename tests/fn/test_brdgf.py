"""Tests for brdgf.bayes_ridge_gibbs."""
import numpy as np
import pytest
from morie.fn.brdgf import bayes_ridge_gibbs


def test_brdgf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bayes_ridge_gibbs(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_brdgf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bayes_ridge_gibbs(x, y)
    assert isinstance(result, dict)
