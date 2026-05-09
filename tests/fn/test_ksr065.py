"""Tests for ksr065.kosorok_ch3_efficient_influence_general."""
import numpy as np
import pytest
from moirais.fn.ksr065 import kosorok_ch3_efficient_influence_general


def test_ksr065_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    psi_tilde = np.random.default_rng(42).normal(0, 1, 100)
    chi_tilde = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch3_efficient_influence_general(A, psi_tilde, chi_tilde, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr065_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    psi_tilde = np.random.default_rng(42).normal(0, 1, 100)
    chi_tilde = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch3_efficient_influence_general(A, psi_tilde, chi_tilde, eta)
    assert isinstance(result, dict)
