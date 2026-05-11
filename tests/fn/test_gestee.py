"""Tests for gestee.gauss_subgaussian_estimator."""
import numpy as np
import pytest
from morie.fn.gestee import gauss_subgaussian_estimator


def test_gestee_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    n = 100
    result = gauss_subgaussian_estimator(y, C, epsilon, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gestee_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    n = 100
    result = gauss_subgaussian_estimator(y, C, epsilon, n)
    assert isinstance(result, dict)
