"""Tests for msm023.mvsml_linear_mixed_models_eq_5_4."""
import numpy as np
import pytest
from morie.fn.msm023 import mvsml_linear_mixed_models_eq_5_4


def test_msm023_basic():
    """Test basic functionality."""
    Another = np.random.default_rng(42).normal(0, 1, 100)
    explored = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    M20 = np.random.default_rng(42).normal(0, 1, 100)
    was = np.random.default_rng(42).normal(0, 1, 100)
    obtained = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(Another, explored, model, M20, was, obtained)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm023_edge():
    """Test edge cases."""
    Another = np.random.default_rng(42).normal(0, 1, 100)
    explored = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    M20 = np.random.default_rng(42).normal(0, 1, 100)
    was = np.random.default_rng(42).normal(0, 1, 100)
    obtained = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(Another, explored, model, M20, was, obtained)
    assert isinstance(result, dict)
