"""Tests for msm032.mvsml_linear_mixed_models_eq_5_6."""
import numpy as np
import pytest
from morie.fn.msm032 import mvsml_linear_mixed_models_eq_5_6


def test_msm032_basic():
    """Test basic functionality."""
    other = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    strategy = np.random.default_rng(42).normal(0, 1, 100)
    In = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    similar = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_6(other, relevant, strategy, In, a, similar)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm032_edge():
    """Test edge cases."""
    other = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    strategy = np.random.default_rng(42).normal(0, 1, 100)
    In = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    similar = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_6(other, relevant, strategy, In, a, similar)
    assert isinstance(result, dict)
