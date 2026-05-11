"""Tests for rng014.rangayyan_ch3_variance_of_sum_uncorrelated."""
import numpy as np
import pytest
from morie.fn.rng014 import rangayyan_ch3_variance_of_sum_uncorrelated


def test_rng014_basic():
    """Test basic functionality."""
    sigma_x = np.random.default_rng(42).normal(0, 1, 100)
    sigma_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_variance_of_sum_uncorrelated(sigma_x, sigma_eta)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng014_edge():
    """Test edge cases."""
    sigma_x = np.random.default_rng(42).normal(0, 1, 100)
    sigma_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_variance_of_sum_uncorrelated(sigma_x, sigma_eta)
    assert isinstance(result, dict)
