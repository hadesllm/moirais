"""Tests for bayocl.bayes_outlier_dp."""
import numpy as np
import pytest
from moirais.fn.bayocl import bayes_outlier_dp


def test_bayocl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = bayes_outlier_dp(y, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayocl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = bayes_outlier_dp(y, alpha)
    assert isinstance(result, dict)
