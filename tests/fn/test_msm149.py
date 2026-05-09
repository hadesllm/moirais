"""Tests for msm149.mvsml_categorical_count_eq_8_8."""
import numpy as np
import pytest
from moirais.fn.msm149 import mvsml_categorical_count_eq_8_8


def test_msm149_basic():
    """Test basic functionality."""
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    substituted = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    Nystr = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(can, be, substituted, the, Q, Nystr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm149_edge():
    """Test edge cases."""
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    substituted = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    Nystr = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(can, be, substituted, the, Q, Nystr)
    assert isinstance(result, dict)
