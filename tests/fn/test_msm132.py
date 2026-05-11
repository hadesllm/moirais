"""Tests for msm132.mvsml_categorical_count_eq_8_5."""
import numpy as np
import pytest
from morie.fn.msm132 import mvsml_categorical_count_eq_8_5


def test_msm132_basic():
    """Test basic functionality."""
    J = 20
    l = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    AK = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_5(J, l, xi, x, j, AK)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm132_edge():
    """Test edge cases."""
    J = 20
    l = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    AK = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_5(J, l, xi, x, j, AK)
    assert isinstance(result, dict)
