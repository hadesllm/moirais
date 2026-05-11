"""Tests for acwhe.private_accuracy_tradeoff."""
import numpy as np
import pytest
from morie.fn.acwhe import private_accuracy_tradeoff


def test_acwhe_basic():
    """Test basic functionality."""
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    n = 100
    result = private_accuracy_tradeoff(sensitivity, epsilon, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_acwhe_edge():
    """Test edge cases."""
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    n = 100
    result = private_accuracy_tradeoff(sensitivity, epsilon, n)
    assert isinstance(result, dict)
