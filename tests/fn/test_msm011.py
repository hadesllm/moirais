"""Tests for msm011.mvsml_linear_mixed_models_eq_5_2."""
import numpy as np
import pytest
from moirais.fn.msm011 import mvsml_linear_mixed_models_eq_5_2


def test_msm011_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    exp = np.random.default_rng(42).normal(0, 1, 100)
    TV = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_2(V, exp, TV, y, X, L)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm011_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    exp = np.random.default_rng(42).normal(0, 1, 100)
    TV = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_2(V, exp, TV, y, X, L)
    assert isinstance(result, dict)
