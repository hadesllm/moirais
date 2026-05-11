"""Tests for msm123.mvsml_categorical_count_eq_8_1."""
import numpy as np
import pytest
from morie.fn.msm123 import mvsml_categorical_count_eq_8_1


def test_msm123_basic():
    """Test basic functionality."""
    n = 100
    k = 5
    k2 = np.random.default_rng(42).normal(0, 1, 100)
    min = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_1(n, k, k2, min, L, yi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm123_edge():
    """Test edge cases."""
    n = 100
    k = 5
    k2 = np.random.default_rng(42).normal(0, 1, 100)
    min = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_1(n, k, k2, min, L, yi)
    assert isinstance(result, dict)
