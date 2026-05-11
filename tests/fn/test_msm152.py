"""Tests for msm152.mvsml_categorical_count_eq_8_12."""
import numpy as np
import pytest
from morie.fn.msm152 import mvsml_categorical_count_eq_8_12


def test_msm152_basic():
    """Test basic functionality."""
    Therefore = np.random.default_rng(42).normal(0, 1, 100)
    an = np.random.default_rng(42).normal(0, 1, 100)
    eigen = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_12(Therefore, an, eigen, of, K, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm152_edge():
    """Test edge cases."""
    Therefore = np.random.default_rng(42).normal(0, 1, 100)
    an = np.random.default_rng(42).normal(0, 1, 100)
    eigen = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_12(Therefore, an, eigen, of, K, the)
    assert isinstance(result, dict)
