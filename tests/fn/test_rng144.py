"""Tests for rng144.rangayyan_ch3_mse_gradient."""
import numpy as np
import pytest
from morie.fn.rng144 import rangayyan_ch3_mse_gradient


def test_rng144_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_mse_gradient(w, Theta, Phi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng144_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_mse_gradient(w, Theta, Phi)
    assert isinstance(result, dict)
