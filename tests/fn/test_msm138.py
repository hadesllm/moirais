"""Tests for msm138.mvsml_categorical_count_eq_8_8."""
import numpy as np
import pytest
from moirais.fn.msm138 import mvsml_categorical_count_eq_8_8


def test_msm138_basic():
    """Test basic functionality."""
    folds = np.random.default_rng(42).normal(0, 1, 100)
    Fig = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    the = np.random.default_rng(42).normal(0, 1, 100)
    optimal = np.random.default_rng(42).normal(0, 1, 100)
    number = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(folds, Fig, d, the, optimal, number)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm138_edge():
    """Test edge cases."""
    folds = np.random.default_rng(42).normal(0, 1, 100)
    Fig = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    the = np.random.default_rng(42).normal(0, 1, 100)
    optimal = np.random.default_rng(42).normal(0, 1, 100)
    number = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(folds, Fig, d, the, optimal, number)
    assert isinstance(result, dict)
