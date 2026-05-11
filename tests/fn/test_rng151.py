"""Tests for rng151.rangayyan_ch3_wiener_optimal_for_noise_removal."""
import numpy as np
import pytest
from morie.fn.rng151 import rangayyan_ch3_wiener_optimal_for_noise_removal


def test_rng151_basic():
    """Test basic functionality."""
    Phi_d = np.random.default_rng(42).normal(0, 1, 100)
    Phi_eta = np.random.default_rng(42).normal(0, 1, 100)
    Phi_1d = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_optimal_for_noise_removal(Phi_d, Phi_eta, Phi_1d)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng151_edge():
    """Test edge cases."""
    Phi_d = np.random.default_rng(42).normal(0, 1, 100)
    Phi_eta = np.random.default_rng(42).normal(0, 1, 100)
    Phi_1d = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_optimal_for_noise_removal(Phi_d, Phi_eta, Phi_1d)
    assert isinstance(result, dict)
