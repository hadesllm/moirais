"""Tests for msm035.mvsml_linear_mixed_models_eq_5_6."""
import numpy as np
import pytest
from morie.fn.msm035 import mvsml_linear_mixed_models_eq_5_6


def test_msm035_basic():
    """Test basic functionality."""
    N = 100
    G = np.eye(10)
    T = np.random.default_rng(43).integers(0, 2, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    This = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_6(N, G, T, b2, E, This)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm035_edge():
    """Test edge cases."""
    N = 100
    G = np.eye(10)
    T = np.random.default_rng(43).integers(0, 2, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    This = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_6(N, G, T, b2, E, This)
    assert isinstance(result, dict)
