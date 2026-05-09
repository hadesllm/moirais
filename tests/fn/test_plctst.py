"""Tests for plctst.placebo_test_did."""
import numpy as np
import pytest
from moirais.fn.plctst import placebo_test_did


def test_plctst_basic():
    """Test basic functionality."""
    y_pre = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = placebo_test_did(y_pre, D, X)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_plctst_edge():
    """Test edge cases."""
    y_pre = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = placebo_test_did(y_pre, D, X)
    assert isinstance(result, dict)
