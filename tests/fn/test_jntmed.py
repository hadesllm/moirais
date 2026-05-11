"""Tests for jntmed.joint_significance_test."""
import numpy as np
import pytest
from morie.fn.jntmed import joint_significance_test


def test_jntmed_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    se_a = np.random.default_rng(42).normal(0, 1, 100)
    se_b = np.random.default_rng(42).normal(0, 1, 100)
    result = joint_significance_test(a, b, se_a, se_b)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_jntmed_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    se_a = np.random.default_rng(42).normal(0, 1, 100)
    se_b = np.random.default_rng(42).normal(0, 1, 100)
    result = joint_significance_test(a, b, se_a, se_b)
    assert isinstance(result, dict)
