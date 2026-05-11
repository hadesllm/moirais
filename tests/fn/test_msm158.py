"""Tests for msm158.mvsml_categorical_count_eq_8_13."""
import numpy as np
import pytest
from morie.fn.msm158 import mvsml_categorical_count_eq_8_13


def test_msm158_basic():
    """Test basic functionality."""
    m = 10
    Reproducing = np.random.default_rng(42).normal(0, 1, 100)
    Kernel = np.random.default_rng(42).normal(0, 1, 100)
    Hilbert = np.random.default_rng(42).normal(0, 1, 100)
    Spaces = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_13(m, Reproducing, Kernel, Hilbert, Spaces, Regression)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm158_edge():
    """Test edge cases."""
    m = 10
    Reproducing = np.random.default_rng(42).normal(0, 1, 100)
    Kernel = np.random.default_rng(42).normal(0, 1, 100)
    Hilbert = np.random.default_rng(42).normal(0, 1, 100)
    Spaces = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_13(m, Reproducing, Kernel, Hilbert, Spaces, Regression)
    assert isinstance(result, dict)
