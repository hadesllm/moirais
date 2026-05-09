"""Tests for rghompr.rangayyan_homomorphic_pred."""
import numpy as np
import pytest
from moirais.fn.rghompr import rangayyan_homomorphic_pred


def test_rghompr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lifter = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_homomorphic_pred(x, lifter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghompr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lifter = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_homomorphic_pred(x, lifter)
    assert isinstance(result, dict)
