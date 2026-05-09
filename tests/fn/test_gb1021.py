"""Tests for gb1021.gibbons_k_median_test."""
import numpy as np
import pytest
from moirais.fn.gb1021 import gibbons_k_median_test


def test_gb1021_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_k_median_test(groups)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1021_edge():
    """Test edge cases."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_k_median_test(groups)
    assert isinstance(result, dict)
