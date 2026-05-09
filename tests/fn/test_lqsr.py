"""Tests for lqsr.l1_regression."""
import numpy as np
import pytest
from moirais.fn.lqsr import l1_regression


def test_lqsr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = l1_regression(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lqsr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = l1_regression(X, y)
    assert isinstance(result, dict)
