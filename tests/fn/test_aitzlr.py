"""Tests for aitzlr.compositional_zero_lrem."""
import numpy as np
import pytest
from moirais.fn.aitzlr import compositional_zero_lrem


def test_aitzlr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_zero_lrem(X, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitzlr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_zero_lrem(X, max_iter)
    assert isinstance(result, dict)
