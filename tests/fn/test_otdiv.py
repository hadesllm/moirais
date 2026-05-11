"""Tests for otdiv.ot_sinkhorn_divergence."""
import numpy as np
import pytest
from morie.fn.otdiv import ot_sinkhorn_divergence


def test_otdiv_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    Cab = np.random.default_rng(42).normal(0, 1, 100)
    Caa = np.random.default_rng(42).normal(0, 1, 100)
    Cbb = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_sinkhorn_divergence(a, b, Cab, Caa, Cbb, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otdiv_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    Cab = np.random.default_rng(42).normal(0, 1, 100)
    Caa = np.random.default_rng(42).normal(0, 1, 100)
    Cbb = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_sinkhorn_divergence(a, b, Cab, Caa, Cbb, epsilon)
    assert isinstance(result, dict)
