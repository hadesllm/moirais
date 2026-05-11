"""Tests for johnsen.johansen_test."""
import numpy as np
import pytest
from morie.fn.johnsen import johansen_test


def test_johnsen_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    det_order = np.random.default_rng(42).normal(0, 1, 100)
    k_ar = np.random.default_rng(42).normal(0, 1, 100)
    result = johansen_test(Y, det_order, k_ar)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_johnsen_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    det_order = np.random.default_rng(42).normal(0, 1, 100)
    k_ar = np.random.default_rng(42).normal(0, 1, 100)
    result = johansen_test(Y, det_order, k_ar)
    assert isinstance(result, dict)
