"""Tests for kmscal.kamath_scaling_laws."""
import numpy as np
import pytest
from morie.fn.kmscal import kamath_scaling_laws


def test_kmscal_basic():
    """Test basic functionality."""
    N = 100
    N_c = np.random.default_rng(42).normal(0, 1, 100)
    alpha_N = np.random.default_rng(42).normal(0, 1, 100)
    L_inf = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_scaling_laws(N, N_c, alpha_N, L_inf)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmscal_edge():
    """Test edge cases."""
    N = 100
    N_c = np.random.default_rng(42).normal(0, 1, 100)
    alpha_N = np.random.default_rng(42).normal(0, 1, 100)
    L_inf = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_scaling_laws(N, N_c, alpha_N, L_inf)
    assert isinstance(result, dict)
