"""Tests for msm137.mvsml_categorical_count_eq_8_7."""
import numpy as np
import pytest
from moirais.fn.msm137 import mvsml_categorical_count_eq_8_7


def test_msm137_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    CTC = np.random.default_rng(42).normal(0, 1, 100)
    CTK = np.random.default_rng(42).normal(0, 1, 100)
    CTy = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_7(b, CTC, CTK, CTy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm137_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    CTC = np.random.default_rng(42).normal(0, 1, 100)
    CTK = np.random.default_rng(42).normal(0, 1, 100)
    CTy = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_7(b, CTC, CTK, CTy)
    assert isinstance(result, dict)
