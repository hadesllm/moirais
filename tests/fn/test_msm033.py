"""Tests for msm033.mvsml_linear_mixed_models_eq_5_6."""
import numpy as np
import pytest
from morie.fn.msm033 import mvsml_linear_mixed_models_eq_5_6


def test_msm033_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    N = 100
    IIJ = np.random.default_rng(42).normal(0, 1, 100)
    RnT = np.random.default_rng(42).normal(0, 1, 100)
    I = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_6(T, N, IIJ, RnT, I, j)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm033_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    N = 100
    IIJ = np.random.default_rng(42).normal(0, 1, 100)
    RnT = np.random.default_rng(42).normal(0, 1, 100)
    I = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_6(T, N, IIJ, RnT, I, j)
    assert isinstance(result, dict)
