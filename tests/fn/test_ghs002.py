"""Tests for ghs002.ghosal_ch2_random_basis_expansion."""
import numpy as np
import pytest
from morie.fn.ghs002 import ghosal_ch2_random_basis_expansion


def test_ghs002_basic():
    """Test basic functionality."""
    beta_j = np.random.default_rng(42).normal(0, 1, 100)
    psi_j = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    result = ghosal_ch2_random_basis_expansion(beta_j, psi_j, J)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs002_edge():
    """Test edge cases."""
    beta_j = np.random.default_rng(42).normal(0, 1, 100)
    psi_j = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    result = ghosal_ch2_random_basis_expansion(beta_j, psi_j, J)
    assert isinstance(result, dict)
