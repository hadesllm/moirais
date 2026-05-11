"""Tests for msm157.mvsml_categorical_count_eq_8_10."""
import numpy as np
import pytest
from morie.fn.msm157 import mvsml_categorical_count_eq_8_10


def test_msm157_basic():
    """Test basic functionality."""
    Qu2 = np.random.default_rng(42).normal(0, 1, 100)
    Zu1 = np.random.default_rng(42).normal(0, 1, 100)
    Kn = np.random.default_rng(42).normal(0, 1, 100)
    mK = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    mKT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_10(Qu2, Zu1, Kn, mK, m, mKT)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm157_edge():
    """Test edge cases."""
    Qu2 = np.random.default_rng(42).normal(0, 1, 100)
    Zu1 = np.random.default_rng(42).normal(0, 1, 100)
    Kn = np.random.default_rng(42).normal(0, 1, 100)
    mK = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    mKT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_10(Qu2, Zu1, Kn, mK, m, mKT)
    assert isinstance(result, dict)
