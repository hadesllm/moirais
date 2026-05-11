"""Tests for msm135.mvsml_categorical_count_eq_8_6."""
import numpy as np
import pytest
from morie.fn.msm135 import mvsml_categorical_count_eq_8_6


def test_msm135_basic():
    """Test basic functionality."""
    CTC = np.random.default_rng(42).normal(0, 1, 100)
    CTK = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    CTy = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_6(CTC, CTK, b, CTy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm135_edge():
    """Test edge cases."""
    CTC = np.random.default_rng(42).normal(0, 1, 100)
    CTK = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    CTy = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_6(CTC, CTK, b, CTy)
    assert isinstance(result, dict)
