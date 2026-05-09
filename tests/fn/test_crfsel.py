"""Tests for crfsel.causal_forest_selection."""
import numpy as np
import pytest
from moirais.fn.crfsel import causal_forest_selection


def test_crfsel_basic():
    """Test basic functionality."""
    forest = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    D = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = causal_forest_selection(forest, X, D, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crfsel_edge():
    """Test edge cases."""
    forest = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    D = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = causal_forest_selection(forest, X, D, y)
    assert isinstance(result, dict)
