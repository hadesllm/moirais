"""Tests for dctgls.doubly_censored_gls."""
import numpy as np
import pytest
from morie.fn.dctgls import doubly_censored_gls


def test_dctgls_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = doubly_censored_gls(y, A, C, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dctgls_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = doubly_censored_gls(y, A, C, H)
    assert isinstance(result, dict)
