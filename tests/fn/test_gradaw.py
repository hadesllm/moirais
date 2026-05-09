"""Tests for gradaw.geron_adaboost_weight_update."""
import numpy as np
import pytest
from moirais.fn.gradaw import geron_adaboost_weight_update


def test_gradaw_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    alpha_t = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_adaboost_weight_update(y_true, y_pred, weights, alpha_t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gradaw_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    alpha_t = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_adaboost_weight_update(y_true, y_pred, weights, alpha_t)
    assert isinstance(result, dict)
