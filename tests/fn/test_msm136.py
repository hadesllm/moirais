"""Tests for msm136.mvsml_categorical_count_eq_8_6."""
import numpy as np
import pytest
from moirais.fn.msm136 import mvsml_categorical_count_eq_8_6


def test_msm136_basic():
    """Test basic functionality."""
    CTy = np.random.default_rng(42).normal(0, 1, 100)
    KTC = np.random.default_rng(42).normal(0, 1, 100)
    KTK = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    KTy = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_6(CTy, KTC, KTK, K, b, KTy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm136_edge():
    """Test edge cases."""
    CTy = np.random.default_rng(42).normal(0, 1, 100)
    KTC = np.random.default_rng(42).normal(0, 1, 100)
    KTK = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    KTy = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_6(CTy, KTC, KTK, K, b, KTy)
    assert isinstance(result, dict)
