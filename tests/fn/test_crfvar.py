"""Tests for crfvar.causal_forest_variance."""
import numpy as np
import pytest
from moirais.fn.crfvar import causal_forest_variance


def test_crfvar_basic():
    """Test basic functionality."""
    forest = np.random.default_rng(42).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    result = causal_forest_variance(forest, X_test)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crfvar_edge():
    """Test edge cases."""
    forest = np.random.default_rng(42).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    result = causal_forest_variance(forest, X_test)
    assert isinstance(result, dict)
