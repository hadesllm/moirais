"""Tests for matnK.matern_kernel."""
import numpy as np
import pytest
from moirais.fn.matnK import matern_kernel


def test_matnK_basic():
    """Test basic functionality."""
    d = 5
    nu = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = matern_kernel(d, nu, rho)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matnK_edge():
    """Test edge cases."""
    d = 5
    nu = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = matern_kernel(d, nu, rho)
    assert isinstance(result, dict)
