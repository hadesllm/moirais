"""Tests for sobtst.sobel_test."""
import numpy as np
import pytest
from moirais.fn.sobtst import sobel_test


def test_sobtst_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    se_a = np.random.default_rng(42).normal(0, 1, 100)
    se_b = np.random.default_rng(42).normal(0, 1, 100)
    result = sobel_test(a, b, se_a, se_b)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_sobtst_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    se_a = np.random.default_rng(42).normal(0, 1, 100)
    se_b = np.random.default_rng(42).normal(0, 1, 100)
    result = sobel_test(a, b, se_a, se_b)
    assert isinstance(result, dict)
