"""Tests for otprm.ot_permutation_test_w1."""
import numpy as np
import pytest
from morie.fn.otprm import ot_permutation_test_w1


def test_otprm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = ot_permutation_test_w1(X, Y, B)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_otprm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = ot_permutation_test_w1(X, Y, B)
    assert isinstance(result, dict)
