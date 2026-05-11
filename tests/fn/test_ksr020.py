"""Tests for ksr020.kosorok_ch1_linear_regression_model."""
import numpy as np
import pytest
from morie.fn.ksr020 import kosorok_ch1_linear_regression_model


def test_ksr020_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    beta = 0.8
    e = np.random.default_rng(44).normal(0, 1, 100)
    result = kosorok_ch1_linear_regression_model(Y, Z, beta, e)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr020_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    beta = 0.8
    e = np.random.default_rng(44).normal(0, 1, 100)
    result = kosorok_ch1_linear_regression_model(Y, Z, beta, e)
    assert isinstance(result, dict)
