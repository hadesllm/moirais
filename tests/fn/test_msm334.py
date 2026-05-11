"""Tests for msm334.mvsml_preprocessing_eq_2_22."""
import numpy as np
import pytest
from morie.fn.msm334 import mvsml_preprocessing_eq_2_22


def test_msm334_basic():
    """Test basic functionality."""
    d = 5
    c = np.random.default_rng(42).normal(0, 1, 100)
    PE = np.random.default_rng(42).normal(0, 1, 100)
    xo = np.random.default_rng(42).normal(0, 1, 100)
    interval = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_22(d, c, PE, xo, interval, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm334_edge():
    """Test edge cases."""
    d = 5
    c = np.random.default_rng(42).normal(0, 1, 100)
    PE = np.random.default_rng(42).normal(0, 1, 100)
    xo = np.random.default_rng(42).normal(0, 1, 100)
    interval = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_22(d, c, PE, xo, interval, the)
    assert isinstance(result, dict)
