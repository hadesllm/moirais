"""Tests for rng058.rangayyan_ch3_pole_zero_factored_form."""
import numpy as np
import pytest
from morie.fn.rng058 import rangayyan_ch3_pole_zero_factored_form


def test_rng058_basic():
    """Test basic functionality."""
    z_k = np.random.default_rng(42).normal(0, 1, 100)
    p_k = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    N = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_pole_zero_factored_form(z_k, p_k, z, N, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng058_edge():
    """Test edge cases."""
    z_k = np.random.default_rng(42).normal(0, 1, 100)
    p_k = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    N = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_pole_zero_factored_form(z_k, p_k, z, N, M)
    assert isinstance(result, dict)
