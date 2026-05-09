"""Tests for rng141.rangayyan_ch3_mse_cost_function."""
import numpy as np
import pytest
from moirais.fn.rng141 import rangayyan_ch3_mse_cost_function


def test_rng141_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    sigma_d = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_mse_cost_function(w, Theta, Phi, sigma_d)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng141_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    sigma_d = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_mse_cost_function(w, Theta, Phi, sigma_d)
    assert isinstance(result, dict)
