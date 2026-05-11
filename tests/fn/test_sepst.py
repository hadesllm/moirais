"""Tests for sepst.separation_set."""
import numpy as np
import pytest
from morie.fn.sepst import separation_set


def test_sepst_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    ci_tests = np.random.default_rng(42).normal(0, 1, 100)
    result = separation_set(X, Y, ci_tests)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sepst_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    ci_tests = np.random.default_rng(42).normal(0, 1, 100)
    result = separation_set(X, Y, ci_tests)
    assert isinstance(result, dict)
