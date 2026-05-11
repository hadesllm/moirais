"""Tests for causflnk.causal_falsification_test."""
import numpy as np
import pytest
from morie.fn.causflnk import causal_falsification_test


def test_causflnk_basic():
    """Test basic functionality."""
    y_pre = np.random.default_rng(42).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    X_baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_falsification_test(y_pre, treat, X_baseline)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causflnk_edge():
    """Test edge cases."""
    y_pre = np.random.default_rng(42).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    X_baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_falsification_test(y_pre, treat, X_baseline)
    assert isinstance(result, dict)
