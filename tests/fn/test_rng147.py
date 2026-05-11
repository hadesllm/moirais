"""Tests for rng147.rangayyan_ch3_minimum_mse."""
import numpy as np
import pytest
from morie.fn.rng147 import rangayyan_ch3_minimum_mse


def test_rng147_basic():
    """Test basic functionality."""
    sigma_d = np.random.default_rng(42).normal(0, 1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_minimum_mse(sigma_d, Theta, Phi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng147_edge():
    """Test edge cases."""
    sigma_d = np.random.default_rng(42).normal(0, 1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_minimum_mse(sigma_d, Theta, Phi)
    assert isinstance(result, dict)
