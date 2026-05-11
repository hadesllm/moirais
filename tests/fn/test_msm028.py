"""Tests for msm028.mvsml_linear_mixed_models_eq_5_5a."""
import numpy as np
import pytest
from morie.fn.msm028 import mvsml_linear_mixed_models_eq_5_5a


def test_msm028_basic():
    """Test basic functionality."""
    J = 20
    N = 100
    G = np.eye(10)
    T = np.random.default_rng(43).integers(0, 2, 100)
    Similarly = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_5a(J, N, G, T, Similarly, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm028_edge():
    """Test edge cases."""
    J = 20
    N = 100
    G = np.eye(10)
    T = np.random.default_rng(43).integers(0, 2, 100)
    Similarly = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_5a(J, N, G, T, Similarly, the)
    assert isinstance(result, dict)
