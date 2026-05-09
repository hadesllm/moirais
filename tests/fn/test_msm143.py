"""Tests for msm143.mvsml_categorical_count_eq_8_9."""
import numpy as np
import pytest
from moirais.fn.msm143 import mvsml_categorical_count_eq_8_9


def test_msm143_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    R2 = np.random.default_rng(42).normal(0, 1, 100)
    When = np.random.default_rng(42).normal(0, 1, 100)
    individuals = np.random.default_rng(42).normal(0, 1, 100)
    had = np.random.default_rng(42).normal(0, 1, 100)
    more = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_9(A, R2, When, individuals, had, more)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm143_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    R2 = np.random.default_rng(42).normal(0, 1, 100)
    When = np.random.default_rng(42).normal(0, 1, 100)
    individuals = np.random.default_rng(42).normal(0, 1, 100)
    had = np.random.default_rng(42).normal(0, 1, 100)
    more = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_9(A, R2, When, individuals, had, more)
    assert isinstance(result, dict)
