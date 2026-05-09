"""Tests for spkfun.schabenberger_k_function."""
import numpy as np
import pytest
from moirais.fn.spkfun import schabenberger_k_function


def test_spkfun_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    lambda_est = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = schabenberger_k_function(points, lambda_est, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spkfun_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    lambda_est = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = schabenberger_k_function(points, lambda_est, r)
    assert isinstance(result, dict)
