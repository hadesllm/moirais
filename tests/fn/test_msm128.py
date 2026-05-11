"""Tests for msm128.mvsml_categorical_count_eq_8_3."""
import numpy as np
import pytest
from morie.fn.msm128 import mvsml_categorical_count_eq_8_3


def test_msm128_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    L = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    kT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_3(X, n, L, yi, kT, i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm128_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    L = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    kT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_3(X, n, L, yi, kT, i)
    assert isinstance(result, dict)
