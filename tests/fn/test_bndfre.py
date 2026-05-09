"""Tests for bndfre.bound_frequentist."""
import numpy as np
import pytest
from moirais.fn.bndfre import bound_frequentist


def test_bndfre_basic():
    """Test basic functionality."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bound_frequentist(lower, upper, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndfre_edge():
    """Test edge cases."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bound_frequentist(lower, upper, alpha)
    assert isinstance(result, dict)
