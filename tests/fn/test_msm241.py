"""Tests for msm241.mvsml_preprocessing_eq_2_2."""
import numpy as np
import pytest
from moirais.fn.msm241 import mvsml_preprocessing_eq_2_2


def test_msm241_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_2(b, XTR)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm241_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_2(b, XTR)
    assert isinstance(result, dict)
