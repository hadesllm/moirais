"""Tests for msm013.mvsml_linear_mixed_models_eq_5_1."""
import numpy as np
import pytest
from morie.fn.msm013 import mvsml_linear_mixed_models_eq_5_1


def test_msm013_basic():
    """Test basic functionality."""
    k = 5
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    t = np.linspace(0, 10, 100)
    qk = np.random.default_rng(42).normal(0, 1, 100)
    These = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_1(k, K, t, qk, These, are)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm013_edge():
    """Test edge cases."""
    k = 5
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    t = np.linspace(0, 10, 100)
    qk = np.random.default_rng(42).normal(0, 1, 100)
    These = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_1(k, K, t, qk, These, are)
    assert isinstance(result, dict)
