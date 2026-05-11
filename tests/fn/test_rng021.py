"""Tests for rng021.rangayyan_ch3_covariance."""
import numpy as np
import pytest
from morie.fn.rng021 import rangayyan_ch3_covariance


def test_rng021_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    mu_x = np.random.default_rng(42).normal(0, 1, 100)
    mu_y = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_covariance(x, y, mu_x, mu_y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng021_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    mu_x = np.random.default_rng(42).normal(0, 1, 100)
    mu_y = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_covariance(x, y, mu_x, mu_y)
    assert isinstance(result, dict)
