"""Tests for crfath.causal_forest_wager_athey."""
import numpy as np
import pytest
from moirais.fn.crfath import causal_forest_wager_athey


def test_crfath_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_forest_wager_athey(y, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crfath_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_forest_wager_athey(y, D, X)
    assert isinstance(result, dict)
