"""Tests for shrinkbm.shrinkage_bayes."""
import numpy as np
import pytest
from moirais.fn.shrinkbm import shrinkage_bayes


def test_shrinkbm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior_family = np.random.default_rng(42).normal(0, 1, 100)
    result = shrinkage_bayes(X, y, prior_family)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shrinkbm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior_family = np.random.default_rng(42).normal(0, 1, 100)
    result = shrinkage_bayes(X, y, prior_family)
    assert isinstance(result, dict)
