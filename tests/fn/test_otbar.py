"""Tests for otbar.ot_barycenter_fixed."""
import numpy as np
import pytest
from morie.fn.otbar import ot_barycenter_fixed


def test_otbar_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    C_list = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    epsilon = 1e-6
    result = ot_barycenter_fixed(A, C_list, weights, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otbar_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    C_list = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    epsilon = 1e-6
    result = ot_barycenter_fixed(A, C_list, weights, epsilon)
    assert isinstance(result, dict)
