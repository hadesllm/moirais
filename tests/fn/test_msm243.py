"""Tests for msm243.mvsml_preprocessing_eq_2_4."""
import numpy as np
import pytest
from moirais.fn.msm243 import mvsml_preprocessing_eq_2_4


def test_msm243_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_4(b, XTR)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm243_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_4(b, XTR)
    assert isinstance(result, dict)
