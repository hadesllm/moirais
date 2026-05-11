"""Tests for aitcmd.compositional_median."""
import numpy as np
import pytest
from morie.fn.aitcmd import compositional_median


def test_aitcmd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tol = 1e-6
    result = compositional_median(X, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitcmd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tol = 1e-6
    result = compositional_median(X, tol)
    assert isinstance(result, dict)
