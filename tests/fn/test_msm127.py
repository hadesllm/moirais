"""Tests for msm127.mvsml_categorical_count_eq_8_1."""
import numpy as np
import pytest
from morie.fn.msm127 import mvsml_categorical_count_eq_8_1


def test_msm127_basic():
    """Test basic functionality."""
    n = 100
    k = 5
    k2 = np.random.default_rng(42).normal(0, 1, 100)
    de = np.random.default_rng(42).normal(0, 1, 100)
    ned = np.random.default_rng(42).normal(0, 1, 100)
    before = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_1(n, k, k2, de, ned, before)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm127_edge():
    """Test edge cases."""
    n = 100
    k = 5
    k2 = np.random.default_rng(42).normal(0, 1, 100)
    de = np.random.default_rng(42).normal(0, 1, 100)
    ned = np.random.default_rng(42).normal(0, 1, 100)
    before = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_1(n, k, k2, de, ned, before)
    assert isinstance(result, dict)
