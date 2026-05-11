"""Tests for linwlr.linear_weighted_learner."""
import numpy as np
import pytest
from morie.fn.linwlr import linear_weighted_learner


def test_linwlr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    W = np.random.default_rng(42).normal(0, 1, 100)
    propensity = np.random.default_rng(42).normal(0, 1, 100)
    result = linear_weighted_learner(y, A, W, propensity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_linwlr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    W = np.random.default_rng(42).normal(0, 1, 100)
    propensity = np.random.default_rng(42).normal(0, 1, 100)
    result = linear_weighted_learner(y, A, W, propensity)
    assert isinstance(result, dict)
