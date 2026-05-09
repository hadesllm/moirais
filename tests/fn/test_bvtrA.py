"""Tests for bvtrA.bias_variance_tradeoff."""
import numpy as np
import pytest
from moirais.fn.bvtrA import bias_variance_tradeoff


def test_bvtrA_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    noise_var = np.random.default_rng(42).normal(0, 1, 100)
    result = bias_variance_tradeoff(y_true, y_pred, noise_var)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bvtrA_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    noise_var = np.random.default_rng(42).normal(0, 1, 100)
    result = bias_variance_tradeoff(y_true, y_pred, noise_var)
    assert isinstance(result, dict)
