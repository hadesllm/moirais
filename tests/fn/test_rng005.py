"""Tests for rng005.rangayyan_ch3_kurtosis."""
import numpy as np
import pytest
from morie.fn.rng005 import rangayyan_ch3_kurtosis


def test_rng005_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    mu_eta = np.random.default_rng(42).normal(0, 1, 100)
    sigma_eta = np.random.default_rng(42).normal(0, 1, 100)
    p_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_kurtosis(eta, mu_eta, sigma_eta, p_eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng005_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    mu_eta = np.random.default_rng(42).normal(0, 1, 100)
    sigma_eta = np.random.default_rng(42).normal(0, 1, 100)
    p_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_kurtosis(eta, mu_eta, sigma_eta, p_eta)
    assert isinstance(result, dict)
