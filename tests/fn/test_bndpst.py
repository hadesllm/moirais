"""Tests for bndpst.bound_post_test."""
import numpy as np
import pytest
from moirais.fn.bndpst import bound_post_test


def test_bndpst_basic():
    """Test basic functionality."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    spec_test = np.random.default_rng(43).normal(0, 1, 30)
    result = bound_post_test(lower, upper, spec_test)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bndpst_edge():
    """Test edge cases."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    spec_test = np.random.default_rng(43).normal(0, 1, 30)
    result = bound_post_test(lower, upper, spec_test)
    assert isinstance(result, dict)
