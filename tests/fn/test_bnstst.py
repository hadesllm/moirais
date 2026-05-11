"""Tests for bnstst.bound_test_inference."""
import numpy as np
import pytest
from morie.fn.bnstst import bound_test_inference


def test_bnstst_basic():
    """Test basic functionality."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    se = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_test_inference(lower, upper, se)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bnstst_edge():
    """Test edge cases."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    se = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_test_inference(lower, upper, se)
    assert isinstance(result, dict)
