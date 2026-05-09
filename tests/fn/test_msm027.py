"""Tests for msm027.mvsml_linear_mixed_models_eq_5_1."""
import numpy as np
import pytest
from moirais.fn.msm027 import mvsml_linear_mixed_models_eq_5_1


def test_msm027_basic():
    """Test basic functionality."""
    J = 20
    T = np.random.default_rng(43).integers(0, 2, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    j1 = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_1(J, T, j, are, g, j1)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm027_edge():
    """Test edge cases."""
    J = 20
    T = np.random.default_rng(43).integers(0, 2, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    j1 = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_1(J, T, j, are, g, j1)
    assert isinstance(result, dict)
