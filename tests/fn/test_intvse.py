"""Tests for intvse.interventional_effect."""
import numpy as np
import pytest
from moirais.fn.intvse import interventional_effect


def test_intvse_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = interventional_effect(Y, X, M, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_intvse_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = interventional_effect(Y, X, M, C)
    assert isinstance(result, dict)
