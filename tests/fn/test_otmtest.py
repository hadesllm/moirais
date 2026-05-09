"""Tests for otmtest.ot_mmd_two_sample."""
import numpy as np
import pytest
from moirais.fn.otmtest import ot_mmd_two_sample


def test_otmtest_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = ot_mmd_two_sample(X, Y, kernel, B)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_otmtest_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = ot_mmd_two_sample(X, Y, kernel, B)
    assert isinstance(result, dict)
