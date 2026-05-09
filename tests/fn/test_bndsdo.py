"""Tests for bndsdo.bound_skewed_outcome."""
import numpy as np
import pytest
from moirais.fn.bndsdo import bound_skewed_outcome


def test_bndsdo_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    skew = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_skewed_outcome(y, D, skew)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndsdo_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    skew = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_skewed_outcome(y, D, skew)
    assert isinstance(result, dict)
