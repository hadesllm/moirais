"""Tests for baylog.bayes_logistic."""
import numpy as np
import pytest
from moirais.fn.baylog import bayes_logistic


def test_baylog_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    prior_scale = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_logistic(y, X, prior_scale)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baylog_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    prior_scale = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_logistic(y, X, prior_scale)
    assert isinstance(result, dict)
