"""Tests for msm153.mvsml_categorical_count_eq_8_12."""
import numpy as np
import pytest
from morie.fn.msm153 import mvsml_categorical_count_eq_8_12


def test_msm153_basic():
    """Test basic functionality."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    eigenvalues = np.random.default_rng(42).normal(0, 1, 100)
    ordered = np.random.default_rng(42).normal(0, 1, 100)
    largest = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    smallest = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_12(the, eigenvalues, ordered, largest, to, smallest)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm153_edge():
    """Test edge cases."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    eigenvalues = np.random.default_rng(42).normal(0, 1, 100)
    ordered = np.random.default_rng(42).normal(0, 1, 100)
    largest = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    smallest = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_12(the, eigenvalues, ordered, largest, to, smallest)
    assert isinstance(result, dict)
