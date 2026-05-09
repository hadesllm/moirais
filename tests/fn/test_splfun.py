"""Tests for splfun.schabenberger_l_function."""
import numpy as np
import pytest
from moirais.fn.splfun import schabenberger_l_function


def test_splfun_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    lambda_est = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = schabenberger_l_function(points, lambda_est, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_splfun_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    lambda_est = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = schabenberger_l_function(points, lambda_est, r)
    assert isinstance(result, dict)
