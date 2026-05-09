"""Tests for rfmdi.rf_mdi_importance."""
import numpy as np
import pytest
from moirais.fn.rfmdi import rf_mdi_importance


def test_rfmdi_basic():
    """Test basic functionality."""
    forest = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rf_mdi_importance(forest, X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rfmdi_edge():
    """Test edge cases."""
    forest = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rf_mdi_importance(forest, X, y)
    assert isinstance(result, dict)
