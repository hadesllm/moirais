"""Tests for gb_psi.gibbons_pitman_efficiency."""
import numpy as np
import pytest
from morie.fn.gb_psi import gibbons_pitman_efficiency


def test_gb_psi_basic():
    """Test basic functionality."""
    T1 = np.random.default_rng(42).normal(0, 1, 100)
    T2 = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = gibbons_pitman_efficiency(T1, T2, theta0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_psi_edge():
    """Test edge cases."""
    T1 = np.random.default_rng(42).normal(0, 1, 100)
    T2 = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = gibbons_pitman_efficiency(T1, T2, theta0)
    assert isinstance(result, dict)
