"""Tests for msm098.mvsml_bayesian_regression_pt2_eq_7_5."""
import numpy as np
import pytest
from moirais.fn.msm098 import mvsml_bayesian_regression_pt2_eq_7_5


def test_msm098_basic():
    """Test basic functionality."""
    pointed = np.random.default_rng(42).normal(0, 1, 100)
    out = np.random.default_rng(42).normal(0, 1, 100)
    Chaps = np.random.default_rng(42).normal(0, 1, 100)
    This = np.random.default_rng(42).normal(0, 1, 100)
    difference = np.random.default_rng(42).normal(0, 1, 100)
    even = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_5(pointed, out, Chaps, This, difference, even)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm098_edge():
    """Test edge cases."""
    pointed = np.random.default_rng(42).normal(0, 1, 100)
    out = np.random.default_rng(42).normal(0, 1, 100)
    Chaps = np.random.default_rng(42).normal(0, 1, 100)
    This = np.random.default_rng(42).normal(0, 1, 100)
    difference = np.random.default_rng(42).normal(0, 1, 100)
    even = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_5(pointed, out, Chaps, This, difference, even)
    assert isinstance(result, dict)
