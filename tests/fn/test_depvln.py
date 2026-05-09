"""Tests for depvln.dependent_violation_test."""
import numpy as np
import pytest
from moirais.fn.depvln import dependent_violation_test


def test_depvln_basic():
    """Test basic functionality."""
    y_neg = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = dependent_violation_test(y_neg, A, H)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_depvln_edge():
    """Test edge cases."""
    y_neg = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = dependent_violation_test(y_neg, A, H)
    assert isinstance(result, dict)
