"""Tests for msm036.mvsml_linear_mixed_models_eq_5_6."""
import numpy as np
import pytest
from morie.fn.msm036 import mvsml_linear_mixed_models_eq_5_6


def test_msm036_basic():
    """Test basic functionality."""
    error = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    Diag = np.random.default_rng(42).normal(0, 1, 100)
    e1 = np.random.default_rng(42).normal(0, 1, 100)
    e2 = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_6(error, R, Diag, e1, e2, The)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm036_edge():
    """Test edge cases."""
    error = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    Diag = np.random.default_rng(42).normal(0, 1, 100)
    e1 = np.random.default_rng(42).normal(0, 1, 100)
    e2 = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_6(error, R, Diag, e1, e2, The)
    assert isinstance(result, dict)
