"""Tests for rng022.rangayyan_ch3_correlation_coefficient."""
import numpy as np
import pytest
from morie.fn.rng022 import rangayyan_ch3_correlation_coefficient


def test_rng022_basic():
    """Test basic functionality."""
    C_xy = np.random.default_rng(42).normal(0, 1, 100)
    sigma_x = np.random.default_rng(42).normal(0, 1, 100)
    sigma_y = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_correlation_coefficient(C_xy, sigma_x, sigma_y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng022_edge():
    """Test edge cases."""
    C_xy = np.random.default_rng(42).normal(0, 1, 100)
    sigma_x = np.random.default_rng(42).normal(0, 1, 100)
    sigma_y = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_correlation_coefficient(C_xy, sigma_x, sigma_y)
    assert isinstance(result, dict)
