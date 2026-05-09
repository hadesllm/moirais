"""Tests for msm129.mvsml_categorical_count_eq_8_3."""
import numpy as np
import pytest
from moirais.fn.msm129 import mvsml_categorical_count_eq_8_3


def test_msm129_basic():
    """Test basic functionality."""
    n = 100
    i = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    k1 = np.random.default_rng(42).normal(0, 1, 100)
    kn = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_3(n, i, where, K, k1, kn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm129_edge():
    """Test edge cases."""
    n = 100
    i = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    k1 = np.random.default_rng(42).normal(0, 1, 100)
    kn = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_3(n, i, where, K, k1, kn)
    assert isinstance(result, dict)
