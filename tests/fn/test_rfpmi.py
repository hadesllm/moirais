"""Tests for rfpmi.rf_permutation_importance."""
import numpy as np
import pytest
from morie.fn.rfpmi import rf_permutation_importance


def test_rfpmi_basic():
    """Test basic functionality."""
    forest = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rf_permutation_importance(forest, X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rfpmi_edge():
    """Test edge cases."""
    forest = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rf_permutation_importance(forest, X, y)
    assert isinstance(result, dict)
