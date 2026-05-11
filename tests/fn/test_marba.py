"""Tests for marba.ma_smd_var_correlated_designs."""
import numpy as np
import pytest
from morie.fn.marba import ma_smd_var_correlated_designs


def test_marba_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    rho = 0.5
    result = ma_smd_var_correlated_designs(g, n, rho)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_marba_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    rho = 0.5
    result = ma_smd_var_correlated_designs(g, n, rho)
    assert isinstance(result, dict)
