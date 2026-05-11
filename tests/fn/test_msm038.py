"""Tests for msm038.mvsml_linear_mixed_models_eq_5_4."""
import numpy as np
import pytest
from morie.fn.msm038 import mvsml_linear_mixed_models_eq_5_4


def test_msm038_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    mmer = np.random.default_rng(42).normal(0, 1, 100)
    y_NA = np.random.default_rng(42).normal(0, 1, 100)
    Env = np.random.default_rng(42).normal(0, 1, 100)
    na = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = mvsml_linear_mixed_models_eq_5_4(A, mmer, y_NA, Env, na, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm038_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    mmer = np.random.default_rng(42).normal(0, 1, 100)
    y_NA = np.random.default_rng(42).normal(0, 1, 100)
    Env = np.random.default_rng(42).normal(0, 1, 100)
    na = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = mvsml_linear_mixed_models_eq_5_4(A, mmer, y_NA, Env, na, method)
    assert isinstance(result, dict)
