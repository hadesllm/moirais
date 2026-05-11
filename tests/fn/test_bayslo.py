"""Tests for bayslo.bayes_lasso."""
import numpy as np
import pytest
from morie.fn.bayslo import bayes_lasso


def test_bayslo_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    lam = 0.1
    result = bayes_lasso(y, M, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayslo_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    lam = 0.1
    result = bayes_lasso(y, M, lam)
    assert isinstance(result, dict)
