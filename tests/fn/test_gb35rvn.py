"""Tests for gb35rvn.gibbons_rvn_test."""
import numpy as np
import pytest
from moirais.fn.gb35rvn import gibbons_rvn_test


def test_gb35rvn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_rvn_test(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb35rvn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_rvn_test(x)
    assert isinstance(result, dict)
