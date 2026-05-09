"""Tests for msm151.mvsml_categorical_count_eq_8_11."""
import numpy as np
import pytest
from moirais.fn.msm151 import mvsml_categorical_count_eq_8_11


def test_msm151_basic():
    """Test basic functionality."""
    m = 10
    US = np.random.default_rng(42).normal(0, 1, 100)
    used = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_11(m, US, used, where, U, are)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm151_edge():
    """Test edge cases."""
    m = 10
    US = np.random.default_rng(42).normal(0, 1, 100)
    used = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_11(m, US, used, where, U, are)
    assert isinstance(result, dict)
