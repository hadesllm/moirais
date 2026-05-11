"""Tests for bndbye.bound_bayes_credible."""
import numpy as np
import pytest
from morie.fn.bndbye import bound_bayes_credible


def test_bndbye_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_bayes_credible(y, X, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndbye_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_bayes_credible(y, X, prior)
    assert isinstance(result, dict)
