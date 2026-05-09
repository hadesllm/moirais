"""Tests for msm130.mvsml_categorical_count_eq_8_3."""
import numpy as np
import pytest
from moirais.fn.msm130 import mvsml_categorical_count_eq_8_3


def test_msm130_basic():
    """Test basic functionality."""
    needs = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    symmetric = np.random.default_rng(42).normal(0, 1, 100)
    positive = np.random.default_rng(42).normal(0, 1, 100)
    semi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_3(needs, to, be, symmetric, positive, semi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm130_edge():
    """Test edge cases."""
    needs = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    symmetric = np.random.default_rng(42).normal(0, 1, 100)
    positive = np.random.default_rng(42).normal(0, 1, 100)
    semi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_3(needs, to, be, symmetric, positive, semi)
    assert isinstance(result, dict)
