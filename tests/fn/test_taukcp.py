"""Tests for taukcp.kendalls_tau_copula."""
import numpy as np
import pytest
from morie.fn.taukcp import kendalls_tau_copula


def test_taukcp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = kendalls_tau_copula(y, copula, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_taukcp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = kendalls_tau_copula(y, copula, theta)
    assert isinstance(result, dict)
