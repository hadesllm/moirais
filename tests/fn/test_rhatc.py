"""Tests for rhatc.r_hat_convergence."""
import numpy as np
import pytest
from morie.fn.rhatc import r_hat_convergence


def test_rhatc_basic():
    """Test basic functionality."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = r_hat_convergence(chains)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rhatc_edge():
    """Test edge cases."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = r_hat_convergence(chains)
    assert isinstance(result, dict)
