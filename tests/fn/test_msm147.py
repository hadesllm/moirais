"""Tests for msm147.mvsml_categorical_count_eq_8_11."""
import numpy as np
import pytest
from moirais.fn.msm147 import mvsml_categorical_count_eq_8_11


def test_msm147_basic():
    """Test basic functionality."""
    Eq = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    eigenvalue = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_11(Eq, the, eigenvalue, of, kernel, matrix)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm147_edge():
    """Test edge cases."""
    Eq = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    eigenvalue = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_11(Eq, the, eigenvalue, of, kernel, matrix)
    assert isinstance(result, dict)
