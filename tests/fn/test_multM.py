"""Tests for multM.multiple_mediators."""
import numpy as np
import pytest
from morie.fn.multM import multiple_mediators


def test_multM_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M_list = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = multiple_mediators(Y, X, M_list, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_multM_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M_list = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = multiple_mediators(Y, X, M_list, C)
    assert isinstance(result, dict)
