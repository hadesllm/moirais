"""Tests for msm103.mvsml_bayesian_regression_pt2_eq_7_3."""
import numpy as np
import pytest
from morie.fn.msm103 import mvsml_bayesian_regression_pt2_eq_7_3


def test_msm103_basic():
    """Test basic functionality."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    ZLg = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_3(L, XE, E, ZLg, e, The)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm103_edge():
    """Test edge cases."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    ZLg = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_3(L, XE, E, ZLg, e, The)
    assert isinstance(result, dict)
