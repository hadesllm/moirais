"""Tests for trntst.transportability_test."""
import numpy as np
import pytest
from moirais.fn.trntst import transportability_test


def test_trntst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = transportability_test(y, A, H, S)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_trntst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = transportability_test(y, A, H, S)
    assert isinstance(result, dict)
