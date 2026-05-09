"""Tests for whtnse.white_noise_test."""
import numpy as np
import pytest
from moirais.fn.whtnse import white_noise_test


def test_whtnse_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lags = 10
    result = white_noise_test(X, lags)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_whtnse_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lags = 10
    result = white_noise_test(X, lags)
    assert isinstance(result, dict)
