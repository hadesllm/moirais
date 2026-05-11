"""Tests for msm144.mvsml_categorical_count_eq_8_10."""
import numpy as np
import pytest
from morie.fn.msm144 import mvsml_categorical_count_eq_8_10


def test_msm144_basic():
    """Test basic functionality."""
    linear = np.random.default_rng(42).normal(0, 1, 100)
    GBLUP = np.random.default_rng(42).normal(0, 1, 100)
    kernels = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    best = np.random.default_rng(42).normal(0, 1, 100)
    second = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_10(linear, GBLUP, kernels, the, best, second)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm144_edge():
    """Test edge cases."""
    linear = np.random.default_rng(42).normal(0, 1, 100)
    GBLUP = np.random.default_rng(42).normal(0, 1, 100)
    kernels = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    best = np.random.default_rng(42).normal(0, 1, 100)
    second = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_10(linear, GBLUP, kernels, the, best, second)
    assert isinstance(result, dict)
