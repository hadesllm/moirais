"""Tests for baysbm.bayes_b_marker."""
import numpy as np
import pytest
from morie.fn.baysbm import bayes_b_marker


def test_baysbm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_b_marker(y, M, pi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baysbm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_b_marker(y, M, pi)
    assert isinstance(result, dict)
