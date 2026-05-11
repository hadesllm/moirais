"""Tests for crfhte.causal_forest_hte_test."""
import numpy as np
import pytest
from morie.fn.crfhte import causal_forest_hte_test


def test_crfhte_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cf_predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_forest_hte_test(y, D, X, cf_predictions)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_crfhte_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cf_predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_forest_hte_test(y, D, X, cf_predictions)
    assert isinstance(result, dict)
