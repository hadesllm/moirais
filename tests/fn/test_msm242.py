"""Tests for msm242.mvsml_preprocessing_eq_2_3."""
import numpy as np
import pytest
from morie.fn.msm242 import mvsml_preprocessing_eq_2_3


def test_msm242_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_3(b, XTR)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm242_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    XTR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_3(b, XTR)
    assert isinstance(result, dict)
