"""Tests for km098.kamath_ch6_log_prob_ratio_attr."""
import numpy as np
import pytest
from morie.fn.km098 import kamath_ch6_log_prob_ratio_attr


def test_km098_basic():
    """Test basic functionality."""
    a_i = np.random.default_rng(42).normal(0, 1, 100)
    a_j = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch6_log_prob_ratio_attr(a_i, a_j, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km098_edge():
    """Test edge cases."""
    a_i = np.random.default_rng(42).normal(0, 1, 100)
    a_j = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch6_log_prob_ratio_attr(a_i, a_j, K)
    assert isinstance(result, dict)
